package ampControl.amp.midi;

import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

import javax.sound.midi.MidiDevice;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.Receiver;
import javax.sound.midi.ShortMessage;

import org.nd4j.linalg.api.ndarray.INDArray;

import ampControl.amp.ClassificationListener;

/**
 * {@link ClassificationListener} which is a MIDI interface capable of sending MIDI commands to a device based on given
 * classification.
 *
 * @author Christian Skärby
 */
public class MidiInterface implements ClassificationListener {


    private final Receiver receiver;
    private final Function<INDArray, List<ShortMessage>> probabilitiesToMessageMapper;

    public MidiInterface(
            Predicate<MidiDevice.Info> wantedDevice,
            Function<INDArray, List<ShortMessage>> probabilitiesToMessageMapper) throws MidiUnavailableException {
        this.probabilitiesToMessageMapper = probabilitiesToMessageMapper;

        // Static methods in MidiSystem -> Not practically testable?? Doesn't really do much so maybe not a big deal
        final MidiDevice.Info midiDeviceInfo = Stream.of(MidiSystem.getMidiDeviceInfo())
                .peek(info -> System.out.println("Found midi device: " + info.getName()))
                .filter(wantedDevice)
                .findAny()
                .orElseThrow(() -> new RuntimeException("MidiDevice not found!"));
        final MidiDevice device = MidiSystem.getMidiDevice(midiDeviceInfo);
        device.open();

        receiver = device.getReceiver();
    }

    @Override
    public void indicateAudioClassification(INDArray probabilities) {
        probabilitiesToMessageMapper.apply(probabilities).forEach(msg -> receiver.send(msg, System.nanoTime()/1000));
    }


    public static void main(String[] args) {
        Stream.of(MidiSystem.getMidiDeviceInfo()).forEach(info -> System.out.println(info.getName() + ": " + info.getDescription()));
    }
}
