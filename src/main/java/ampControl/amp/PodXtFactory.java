package ampControl.amp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.ShortMessage;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

import ampControl.admin.param.IntToDoubleConverter;
import ampControl.admin.param.PodXtProgramChangeStringConverter;
import ampControl.amp.labelmapping.LabelMapping;
import ampControl.amp.labelmapping.MaskDuplicateLabelMapping;
import ampControl.amp.labelmapping.MaskingLabelMapping;
import ampControl.amp.labelmapping.MidiProgramChangeLabelMapping;
import ampControl.amp.labelmapping.MomentumLabelMapping;
import ampControl.amp.labelmapping.PacingLabelMapping;
import ampControl.amp.midi.Devices;
import ampControl.amp.midi.MidiInterface;
import ampControl.amp.midi.program.PodXtProgramChange;
import ampControl.amp.midi.program.ProgramChange;
import ampControl.amp.probabilities.ArgMax;
import ampControl.amp.probabilities.Interpreter;
import ampControl.amp.probabilities.ThresholdFilter;

/**
 * Factory for creating a {@link ClassificationListener} which sets the program on a PodXt over midi based on
 * label probabilities.
 * Only PodXt specific thing about this class is really the mapping between {@link PodXtProgramChange} and Midi
 * messages and the hardcoded midi device to use.
 * TODO: Refactor this class to separate PodXt specific stuff from generic MIDI program change stuff.
 *
 * @author Christian Skärby
 *
 */
@Parameters(commandDescription = "Sets up POD XT for automatic switching")
public class PodXtFactory implements ClassificationListener.Factory {

    @Parameter(names = "-midiChannel", description = "midi channel to use")
    private int midiChannel = 0;

    @Parameter(names = {"-labelToProg", "-ltp"},
            description = "Comma separated list of how to map labels programs. First program is mapped to label 0 etc",
            converter = PodXtProgramChangeStringConverter.class)
    private List<PodXtProgramChange> programChangesList = Arrays.asList(
            PodXtProgramChange.D21, // Silence
            PodXtProgramChange.A17, // Noise
            PodXtProgramChange.A17, // Rythm
            PodXtProgramChange.B17  // Lead
    );

    @Parameter(names = {"-labelToThreshold", "-ltt"},
            description = "Comma separated list of how to map labels programs. Syntax is <labelx>:<thredholsx>,<labely>:<thresholdy,...>",
            converter = IntToDoubleConverter.class)
    private Map<Integer, Double> probabilityThresholds = new IntToDoubleConverter().convert("" +
            "0:0.8," +
            "1:0.8," +
            "2:0.95," +
            "3:0.85");

    @Parameter(names = {"-labelMask", "-lm"}, description = "Comma separated list of labels which shall not be reacted to")
    List<Integer> labelsMask = Arrays.asList(0, 1);

    @Parameter(names = {"-switchThreshold", "-st"}, description = "How many consecutive identical classifications are needed to switch")
    private int switchMomentumThreshold = 3;

    @Parameter(names = {"-updateProhibitTime", "-upt"}, description = "Shortest time in milli seconds between program changes allowed")
    private int updateProhibitTimeMs = 500;

    @Override
    public ClassificationListener create() {
        Function<INDArray, List<ShortMessage>> probabilitiesToMessageMapping = getProbabilitiesToMessageMapping();

        try {
            return new MidiInterface(Devices.podXt, probabilitiesToMessageMapping);
        } catch (MidiUnavailableException e) {
            throw new RuntimeException("Midi device initialization failed!", e);
        }
    }

    /**
     * Create a mapping between probabilities for each label and the midi message to be sent
     *
     * @return a mapping between probabilities for each label and the midi message to be sent
     */
    Function<INDArray, List<ShortMessage>> getProbabilitiesToMessageMapping() {
        Interpreter<Integer> interpretProbabilities = new ArgMax();
        for (Map.Entry<Integer, Double> labelThreshEntry : probabilityThresholds.entrySet()) {
            interpretProbabilities = new ThresholdFilter<>(labelThreshEntry.getKey(), labelThreshEntry.getValue(), interpretProbabilities);
        }

        // Note that order matters alot here since most label mappings are stateful. Any ideas on how to realize the same
        // thing without being stateful are more than welcome
        final LabelMapping<ShortMessage> mapLabelToProgram =
                new MomentumLabelMapping<>(switchMomentumThreshold,
                        new MaskDuplicateLabelMapping<>(
                                new PacingLabelMapping<>(updateProhibitTimeMs,
                                        new MaskingLabelMapping<>(labelsMask,
                                                new MidiProgramChangeLabelMapping(midiChannel, programChangesList.toArray(new ProgramChange[]{})
                                                )
                                        )
                                )
                        )
                );

        final int invalidLabel = 6666;
        return interpretProbabilities
                .andThen(list -> {
                    if (!list.isEmpty()) {
                        return list.get(0);
                    }
                    return invalidLabel;
                })
                .andThen(new MaskingLabelMapping<>(Collections.singletonList(invalidLabel), mapLabelToProgram));
    }

    public static void main(String[] args) {
        PodXtFactory fac = new PodXtFactory();
        JCommander.newBuilder().addObject(fac).build().parse(args);

        fac.create().indicateAudioClassification(Nd4j.create(new double[]{1, 0, 0, 0}));
    }
}
