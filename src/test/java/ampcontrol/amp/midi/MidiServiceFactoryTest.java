package ampcontrol.amp.midi;

import ampcontrol.admin.service.MockControlRegistry;
import ampcontrol.admin.service.Service;
import com.beust.jcommander.JCommander;
import org.junit.Test;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.ShortMessage;
import java.util.function.Consumer;

import static org.junit.Assert.*;

/**
 * Test cases for {@link MidiServiceFactory}.
 *
 * @author Christian Skärby
 */
public class MidiServiceFactoryTest {

    private final static String topicPar = "-midiTopic";
    private final static String topicStr = "grergre/hthjj/rere";

    /**
     * Test that a correct message is delivered to the receiver
     */
    @Test
    public void receiveMessage() {
        final int channel = 3;
        final int command = ShortMessage.CONTROL_CHANGE;
        final int data1 = 66;
        final int data2 = 77;

        final MidiServiceFactory factory = new MidiServiceFactory(()-> channel);

        final String parString = topicPar + " " + topicStr;
        JCommander.newBuilder().addObject(factory).build()
                .parse(parString.split(" "));

        try {
            final ProbeConsumer probe = new ProbeConsumer();
            final Service service = factory.createService(probe);
            final MockControlRegistry registry = new MockControlRegistry();

            service.registerTo(registry);

            registry.deliver(topicStr, command + "," + data1 + "," + data2);

            final ShortMessage expected = new ShortMessage(command, channel, data1, data2);
            probe.assertMessage(expected);

        } catch (InvalidMidiDataException e) {
            fail("Could not create expected midi message!" + e);
        }
    }

    /**
     * Test that an incorrect message is not delivered to the receiver
     */
    @Test
    public void incorrectMessage() {
        final int channel = 666;
        final int command = 99999;
        final int data1 = Integer.MAX_VALUE;
        final int data2 = Integer.MIN_VALUE;

        final MidiServiceFactory factory = new MidiServiceFactory(() -> channel);

        final String parString = topicPar + " " + topicStr;
        JCommander.newBuilder().addObject(factory).build()
                .parse(parString.split(" "));
        final ProbeConsumer probe = new ProbeConsumer();
        final Service service = factory.createService(probe);
        final MockControlRegistry registry = new MockControlRegistry();

        service.registerTo(registry);

        registry.deliver(topicStr, command + "," + data1 + "," + data2);

        probe.assertMessage(null);
    }

    private static final class ProbeConsumer implements Consumer<ShortMessage> {

        private ShortMessage actual;

        private void assertMessage(ShortMessage expected) {
            if (expected == null) {
                assertNull("Expected no message!", actual);
            } else {
                assertNotNull("No message received!", actual);
                assertEquals("Incorrect channel!", expected.getCommand(), actual.getCommand());
                assertEquals("Incorrect channel!", expected.getChannel(), actual.getChannel());
                assertEquals("Incorrect data1!", expected.getData1(), actual.getData1());
                assertEquals("Incorrect data2!", expected.getData2(), actual.getData2());
            }
        }

        @Override
        public void accept(ShortMessage shortMessage) {
            actual = shortMessage;
        }
    }
}