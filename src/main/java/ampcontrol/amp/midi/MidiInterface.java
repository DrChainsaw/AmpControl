package ampcontrol.amp.midi;

import ampcontrol.admin.service.Service;
import ampcontrol.admin.service.control.SubscriptionRegistry;
import ampcontrol.amp.AmpInterface;
import ampcontrol.amp.ClassificationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.sound.midi.*;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * {@link ClassificationListener} which is a MIDI interface capable of sending MIDI commands to a device based on given
 * classification.
 *
 * @author Christian Skärby
 */
public class MidiInterface implements AmpInterface {


    private final Receiver receiver;
    private final Service serviceDelegate;
    private final Function<INDArray, List<ShortMessage>> probabilitiesToMessageMapper;

    MidiInterface(
            Predicate<MidiDevice.Info> wantedDevice,
            Function<INDArray, List<ShortMessage>> probabilitiesToMessageMapper,
            Function<Receiver, Service> serviceFactory
    ) throws MidiUnavailableException {
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
        serviceDelegate = serviceFactory.apply(receiver);
    }

    @Override
    public void indicateAudioClassification(INDArray probabilities) {
        probabilitiesToMessageMapper.apply(probabilities).forEach(msg -> receiver.send(msg, System.nanoTime() / 1000));
    }

    @Override
    public void stop() {
        serviceDelegate.stop();
    }

    @Override
    public void registerTo(SubscriptionRegistry subscriptionRegistry) {
        serviceDelegate.registerTo(subscriptionRegistry);
    }

    public static void main(String[] args) {
        Stream.of(MidiSystem.getMidiDeviceInfo()).forEach(info -> System.out.println(info.getName() + ": " + info.getDescription()));
    }
}
