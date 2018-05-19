package ampcontrol.amp.midi;

import ampcontrol.admin.param.StringToMidiDevicePredicateConverter;
import ampcontrol.admin.service.NoService;
import ampcontrol.amp.AmpInterface;
import ampcontrol.amp.midi.program.MidiProgramChange;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParametersDelegate;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.sound.midi.MidiDevice;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.ShortMessage;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Factory for creating a {@link AmpInterface} which sets the program on a generic MIDI device based on
 * label probabilities.
 *
 * @author Christian Skärby
 *
 */
public class MidiProgChangeFactory implements AmpInterface.Factory {

    @Parameter(names = {"-midiDevice, -md"}, description = "Which midi device to send program change to based on classification",
            converter = StringToMidiDevicePredicateConverter.class)
    private Predicate<MidiDevice.Info> device = Devices.audioBox44Vsl;

    @Parameter(names = {"-labelToProg", "-ltp"},
            description = "Comma separated list of how to map labels programs. First program is mapped to label 0 etc")
    private List<Integer> programChangesList = Arrays.asList(
            2, // Silence
            2, // Noise
            2, // Rythm
            3  // Lead
    );

    @ParametersDelegate
    private final MidiChannelPar midiChannelPar = new MidiChannelPar();

    @ParametersDelegate
    private final ProbabilitiesToMidiProgramChange programChange = new ProbabilitiesToMidiProgramChange(midiChannelPar);

    @Override
    public AmpInterface create() {
        Function<INDArray, List<ShortMessage>> probabilitiesToMessageMapping = programChange.createProbabilitiesToMessageMapping(
                programChangesList.stream()
                        .map(p -> MidiProgramChange.valueOf("p" + p))
                        .collect(Collectors.toList()));

        try {
            return new MidiInterface(device, probabilitiesToMessageMapping, rec -> new NoService());
        } catch (MidiUnavailableException e) {
            throw new IllegalStateException("Midi device initialization failed!", e);
        }
    }
}
