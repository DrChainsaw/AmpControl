package ampControl.amp;

import ampControl.admin.param.PodXtProgramChangeStringConverter;
import ampControl.amp.midi.Devices;
import ampControl.amp.midi.MidiInterface;
import ampControl.amp.midi.program.PodXtProgramChange;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.ShortMessage;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

/**
 * Factory for creating a {@link ClassificationListener} which sets the program on a PodXt over midi based on
 * label probabilities.
 *
 * @author Christian Skärby
 *
 */
@Parameters(commandDescription = "Sets up POD XT for automatic switching")
public class PodXtFactory implements ClassificationListener.Factory {

    @Parameter(names = {"-labelToProg", "-ltp"},
            description = "Comma separated list of how to map labels programs. First program is mapped to label 0 etc",
            converter = PodXtProgramChangeStringConverter.class)
    private List<PodXtProgramChange> programChangesList = Arrays.asList(
            PodXtProgramChange.D21, // Silence
            PodXtProgramChange.A17, // Noise
            PodXtProgramChange.A17, // Rythm
            PodXtProgramChange.B17  // Lead
    );

    @ParametersDelegate
    ProbabilitiesToMidiProgramChange programChange = new ProbabilitiesToMidiProgramChange();

    @Override
    public ClassificationListener create() {
        Function<INDArray, List<ShortMessage>> probabilitiesToMessageMapping = programChange.createProbabilitiesToMessageMapping(programChangesList);

        try {
            return new MidiInterface(Devices.podXt, probabilitiesToMessageMapping);
        } catch (MidiUnavailableException e) {
            throw new RuntimeException("Midi device initialization failed!", e);
        }
    }

    public static void main(String[] args) {
        PodXtFactory fac = new PodXtFactory();
        JCommander.newBuilder().addObject(fac).build().parse(args);

        fac.create().indicateAudioClassification(Nd4j.create(new double[]{1, 0, 0, 0}));
    }
}
