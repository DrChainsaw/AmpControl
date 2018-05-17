package ampcontrol.amp.midi;

import javax.sound.midi.MidiDevice;
import java.util.Optional;
import java.util.function.Predicate;

/**
 * Static class with pre-defined predicates to filter out wanted devices from a list of found MIDI devices.
 *
 * @author Christian Skärby
 */
public class Devices {

    /**
     * Matches POD XT midi interface.
     */
    public static final Predicate<MidiDevice.Info> podXt =
            info -> info.getName().matches(".*Line 6 PODxt") && info.getDescription().equals("External MIDI Port");

    /**
     * Matches AudioBox 44VSL midi out
     */
    public static final Predicate<MidiDevice.Info> audioBox44Vsl =
            info -> info.getName().matches("AudioBox 44VSL MIDI Out");

    /**
     * String accessor for command line arguments
     * @param str
     * @return Optional containing the sought predicate, empty if no match
     */
    public static Optional<Predicate<MidiDevice.Info>> getPredefined(String str) {
        switch (str.toLowerCase()) {
            case "podxt":
                return Optional.of(podXt);
            case "audiobox44vsl":
                return Optional.of(audioBox44Vsl);
            default:
                return Optional.empty();

        }
    }

}
