package ampControl.amp.midi;

import java.util.function.Predicate;

import javax.sound.midi.MidiDevice;

/**
 * Static class with pre-defined predictes to filter out wanted devices from a list of found MIDI devices.
 *
 * @author Christian Skärby
 */
public class Devices {

    /**
     * Matches POD XT midi interface.
     */
    public static final Predicate<MidiDevice.Info> podXt =
            info -> info.getName().matches(".*Line 6 PODxt") && info.getDescription().equals("External MIDI Port");

}
