package ampControl.admin.param;

import com.beust.jcommander.ParameterException;
import ampControl.amp.midi.program.PodXtProgramChange;
import ampControl.amp.midi.program.ProgramChange;
import org.junit.Test;

import static org.junit.Assert.*;

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