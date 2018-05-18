package ampcontrol.admin.param;

import com.beust.jcommander.ParameterException;
import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link IntToDoubleConverter}
 *
 * @author Christian Skärby
 */
public class IntToDoubleConverterTest {

    /**
     * Test a valid conversion
     */
    @Test
    public void convertValid() {
        final Map<Integer, Double> expected = new LinkedHashMap<>();
        expected.put(0, 1.2);
        expected.put(3, 4.5);
        expected.put(6, 7.89);

        final String testStr = expected.entrySet().stream()
                .map(entry -> entry.getKey() + ":" + entry.getValue())
                .collect(Collectors.joining(","));

        assertEquals("Incorrect parsing!", expected, new IntToDoubleConverter().convert(testStr));
    }

    /**
     * Test that empty string results in an exception (should it?)
     */
    @Test(expected = ParameterException.class)
    public void convertEmpty() {
        new IntToDoubleConverter().convert("");
    }

    /**
     * Test that key without value results in an exception
     */
    @Test(expected = ParameterException.class)
    public void convertMissingValue() {
        new IntToDoubleConverter().convert("0:1.2,3");
    }

    /**
     * Test that non integer key results in an exception
     */
    @Test(expected = ParameterException.class)
    public void convertNonIntegerKey() {
        new IntToDoubleConverter().convert("0.2:1.2");
    }

    /**
     * Test that non-double value results in an exception
     */
    @Test(expected = ParameterException.class)
    public void convertNonDoubleValue() {
        new IntToDoubleConverter().convert("0.2:1.a");
    }

}