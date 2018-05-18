package ampcontrol.admin.param;

import ampcontrol.amp.midi.program.PodXtProgramChange;
import ampcontrol.amp.midi.program.ProgramChange;
import com.beust.jcommander.ParameterException;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link PodXtProgramChangeStringConverter}
 *
 * @author Christian Skärby
 */
public class PodXtProgramChangeStringConverterTest {

    /**
     * Test valid input
     */
    @Test
    public void convertValid() {
        ProgramChange expected = PodXtProgramChange.A17;
        assertEquals("Incorrect mapping!", expected, new PodXtProgramChangeStringConverter().convert(expected.toString()));
    }

    /**
     * Test invalid input
     */
    @Test(expected = ParameterException.class)
    public void convertInvalid() {
    new PodXtProgramChangeStringConverter().convert("LOREM IPSUM...");
    }
}