package ampcontrol.amp.midi;

import com.beust.jcommander.Parameter;

import java.util.function.IntSupplier;

/**
 * Midi channel from JCommander. Separate class so that the same parameter can be used in many places to reduce the risk
 * of inconistency.
 *
 * @author Christian Skärby
 */
public class MidiChannelPar implements IntSupplier {

    @Parameter(names = "-midiChannel", description = "midi channel to use")
    private int midiChannel = 0;

    @Override
    public int getAsInt() {
        return midiChannel;
    }

}
