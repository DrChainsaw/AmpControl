package ampcontrol.admin.param;

import ampcontrol.amp.midi.Devices;
import com.beust.jcommander.IStringConverter;
import org.junit.Test;

import javax.sound.midi.MidiDevice;
import java.util.function.Predicate;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link StringToMidiDevicePredicateConverter}
 *
 * @author Christian Skärby
 */
public class StringToMidiDevicePredicateConverterTest {

    /**
     * Test conversion of a string to a predefined midi interface
     */
    @Test
    public void convertPredefined() {
        final IStringConverter<Predicate<MidiDevice.Info>> converter = new StringToMidiDevicePredicateConverter();
        assertEquals("Incorrect conversion", Devices.podXt, converter.convert("PodXT"));
        assertEquals("Incorrect conversion", Devices.podXt, converter.convert("podxt"));
        assertEquals("Incorrect conversion", Devices.audioBox44Vsl, converter.convert("AudioBox44Vsl"));
        assertEquals("Incorrect conversion", Devices.audioBox44Vsl, converter.convert("audiobox44vsl"));
    }
}