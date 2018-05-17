package ampcontrol.amp.midi;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.ShortMessage;

/**
 * Creates a {@link javax.sound.midi.ShortMessage} from a String. Basically a hand-rolled deserialization protocol. The
 *  expected format of cmdData1Data2 is "command,data1,data2" where command, data1 and data2 are integer numbers.
 *
 * @author Christian Skärby
 */
class MidiMessageFromString {

    private final ShortMessage msg;

    /**
     * Constructor
     * @param channel the midi channel
     * @param cmdData1Data2 expected format is "command,data1,data2" where command, data1 and data2 are integer numbers
     * @throws InvalidMidiDataException in case a ShortMessage could not be built
     */
    MidiMessageFromString(int channel, String cmdData1Data2) throws InvalidMidiDataException {
        String[] cmdData1Data2Arr = cmdData1Data2.split(",");
        if (cmdData1Data2Arr.length != 3) {
            throw new IllegalArgumentException("Incorrect lenght of message " + cmdData1Data2);
        }
        msg = new ShortMessage(
                Integer.parseInt(cmdData1Data2Arr[0]),
                channel,
                Integer.parseInt(cmdData1Data2Arr[1]),
                Integer.parseInt(cmdData1Data2Arr[2]));
    }

    public ShortMessage get() {
        return msg;
    }

}
