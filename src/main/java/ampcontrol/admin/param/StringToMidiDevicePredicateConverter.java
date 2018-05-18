package ampcontrol.admin.param;

import ampcontrol.amp.midi.Devices;
import com.beust.jcommander.IStringConverter;

import javax.sound.midi.MidiDevice;
import java.util.function.Predicate;

/**
 * Converts a String into a {@link Predicate<MidiDevice.Info>}.
 *
 * @author Christian Skärby
 */
public class StringToMidiDevicePredicateConverter implements IStringConverter<Predicate<MidiDevice.Info>> {
    @Override
    public Predicate<MidiDevice.Info> convert(String s) {
        return Devices.getPredefined(s).orElseThrow(() -> new IllegalArgumentException("No predefined midi device predicate found for " + s));
        // TODO: Parse s into predicates, e.g. name=blala,desc=thisandthat
    }
}
