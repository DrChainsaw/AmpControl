package ampControl.amp.midi;

import com.beust.jcommander.Parameter;

/**
 * Midi channel from JCommander. Separate class so that the same parameter can be used in many places without risk
 * of inconistency.
 *
 * @author Christian Skärby
 */
public class MidiChannelPar {

    @Parameter(names = "-midiChannel", description = "midi channel to use")
    private int midiChannel = 0;

    int get() {
        return midiChannel;
    }

}
