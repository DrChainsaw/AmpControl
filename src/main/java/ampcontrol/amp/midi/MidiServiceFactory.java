package ampcontrol.amp.midi;

import ampcontrol.admin.service.Service;
import ampcontrol.admin.service.control.SubscriptionRegistry;
import com.beust.jcommander.Parameter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.ShortMessage;
import java.util.function.Consumer;
import java.util.function.IntSupplier;

/**
 * Configuration of a {@link Service} which subscribes to a topic and interprets any messages as a
 * {@link MidiMessageFromString}.
 *
 * @author Christian Skärby
 */
public class MidiServiceFactory {

    @Parameter(names = "-midiTopic", description = "Topic to listen to midi commands. Format is cmd,data1,data2 all " +
            "integers")
    private String topic = "midi/message";

    private final IntSupplier midiChannel;

    public MidiServiceFactory(IntSupplier midiChannel) {
        this.midiChannel = midiChannel;
    }

    private static class ServiceInternal implements Service {

        private static final Logger log = LoggerFactory.getLogger(MidiServiceFactory.ServiceInternal.class);


        private final Consumer<ShortMessage> listener;
        private final int channel;
        private final String topic;
        private boolean on = true;

        private ServiceInternal(Consumer<ShortMessage> listener, int channel, String topic) {
            this.listener = listener;
            this.channel = channel;
            this.topic = topic;
        }

        @Override
        public void stop() {
            on = false;
        }

        @Override
        public void registerTo(SubscriptionRegistry subscriptionRegistry) {
            subscriptionRegistry.registerSubscription(topic, msg -> {
                if(on) {
                    try {
                        listener.accept(new MidiMessageFromString(channel, msg).get());
                    } catch (InvalidMidiDataException e) {
                        log.warn("Incorrect MIDI message: " + e);
                    }
                }
            });
        }
    }

    /**
     * Create a {@link Service} which sends {@link ShortMessage ShortMessages} received on the configured topic
     * to the given Consumer.
     *
     * @param listener Consumer of ShortMessages
     * @return a {@link Service}
     */
    public Service createService(Consumer<ShortMessage> listener) {
        return new ServiceInternal(listener, midiChannel.getAsInt(), topic);
    }
}
