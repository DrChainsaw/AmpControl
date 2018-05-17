package ampcontrol.amp.midi;

import ampcontrol.admin.param.IntToDoubleConverter;
import ampcontrol.amp.labelmapping.*;
import ampcontrol.amp.midi.program.ProgramChange;
import ampcontrol.amp.probabilities.ArgMax;
import ampcontrol.amp.probabilities.Interpreter;
import ampcontrol.amp.probabilities.ThresholdFilter;
import com.beust.jcommander.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.sound.midi.ShortMessage;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.IntSupplier;

/**
 * Generic parameterized class for mapping a probabilties of each label into a midi program change. Sets up heuristics
 * to mitigate toggling between programs at the cost of switching delay.
 *
 * @author Christian Skärby
 */
class ProbabilitiesToMidiProgramChange {

    @Parameter(names = {"-labelToThreshold", "-ltt"},
            description = "Comma separated list of how to map labels programs. Syntax is <labelx>:<thredholsx>,<labely>:<thresholdy,...>",
            converter = IntToDoubleConverter.class)
    private Map<Integer, Double> probabilityThresholds = new IntToDoubleConverter().convert("" +
            "0:0.8," +
            "1:0.8," +
            "2:0.95," +
            "3:0.85");

    @Parameter(names = {"-labelMask", "-lm"}, description = "Comma separated list of labels which shall not be reacted to")
    private List<Integer> labelsMask = Arrays.asList(0, 1);

    @Parameter(names = {"-switchThreshold", "-st"}, description = "How many consecutive identical classifications are needed to switch")
    private int switchMomentumThreshold = 3;

    @Parameter(names = {"-updateProhibitTime", "-upt"}, description = "Shortest time in milli seconds between program changes allowed")
    private int updateProhibitTimeMs = 500;

    private final IntSupplier midiChannel;

    ProbabilitiesToMidiProgramChange(IntSupplier midiChannel) {
        this.midiChannel = midiChannel;
    }

    /**
     * Create a mapping between probabilities for each label and the midi message to be sent
     *
     * @return a mapping between probabilities for each label and the midi message to be sent
     */
    Function<INDArray, List<ShortMessage>> createProbabilitiesToMessageMapping(List<? extends ProgramChange> programChangesList) {
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
                                                new MidiProgramChangeLabelMapping(midiChannel.getAsInt(), programChangesList.toArray(new ProgramChange[]{})
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
}
