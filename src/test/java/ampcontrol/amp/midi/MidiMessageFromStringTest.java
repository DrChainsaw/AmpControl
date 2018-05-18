package ampcontrol.amp.midi;

import org.junit.Test;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.ShortMessage;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test cases for {@link MidiMessageFromString}
 *
 * @author Christian Skärby
 */
public class MidiMessageFromStringTest {

    private final static String cmdPar = "-str2midCmd";
    private final static String d1Par = "-str2midD1";
    private final static String d2Par = "-str2midD2";

    private final static String cmdVal = "cmd";
    private final static String d1Val = "d1";
    private final static String d2Val = "d2";

    /**
     * Test that message can be created
     */
    @Test
    public void get() {
        final int cmd = ShortMessage.CONTROL_CHANGE;
        final int channel = 11;
        final int data1 = 33;
        final int data2 = 77;

        try {
            final ShortMessage expected = new ShortMessage(ShortMessage.CONTROL_CHANGE, channel, data1, data2);
            final String msg = cmd + "," + data1 + ","  + data2;
            final ShortMessage actual = new MidiMessageFromString(channel, msg).get();

            assertEquals("Incorrect channel!", expected.getCommand(), actual.getCommand());
            assertEquals("Incorrect channel!", expected.getChannel(), actual.getChannel());
            assertEquals("Incorrect data1!", expected.getData1(), actual.getData1());
            assertEquals("Incorrect data2!", expected.getData2(), actual.getData2());
        } catch (InvalidMidiDataException e) {
            fail("Could not create MIDI message!" + e);
        }
    }
}