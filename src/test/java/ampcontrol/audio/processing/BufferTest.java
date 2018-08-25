package ampcontrol.audio.processing;

import org.junit.Test;

import java.util.Optional;

import static org.junit.Assert.*;

/**
 * Test cases for {@link Buffer}
 *
 * @author Christian Skärby
 */
public class BufferTest {

    /**
     * Test that name is correct
     */
    @Test
    public void name() {
        assertEquals("Incorrect name!", Buffer.nameStatic(), new Buffer().name());
    }

    @Test
    public void create() {
        final double[][] inputArr = {{0, 1, 2}, {3, 4, 5}}; // Will be changed later to verify that input really was buffered
        final double[][] expected = {{0, 1, 2}, {3, 4, 5}}; // This is always the expected result
        final ProcessingResult input = new SingletonDoubleInput(inputArr);
        final ProcessingResult.Factory buffer = new Buffer();
        ProcessingResult output = buffer.create(input);

        assertOutputExistsAndEqualTo(expected, output);

        // Tamper with input just to verify that it was buffered. Not expected to be normal operation
        inputArr[0][0] = 666;
        // Testcase would be meaningless if input would not change as a result of the above
        assertOutputExistsAndEqualTo(inputArr, input);

        double[][] out = assertOutputExistsAndEqualTo(expected, output);

        // Tamper with output to verify that this does not corrupt buffer
        out[0][0] = out[0][0] + 111;
        assertOutputExistsAndEqualTo(expected, output);
    }

    private double[][] assertOutputExistsAndEqualTo(double[][] expected, ProcessingResult result) {
        final Optional<double[][]> outOpt = result.stream().findFirst();
        assertTrue("Expected result!", outOpt.isPresent());
        assertArrayEquals("Incorrect output!", expected, outOpt.get());
        return outOpt.get();
    }
}