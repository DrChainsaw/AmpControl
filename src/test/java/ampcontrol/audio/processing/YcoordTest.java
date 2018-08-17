package ampcontrol.audio.processing;

import org.junit.Test;

import java.util.Optional;

import static org.junit.Assert.*;

/**
 * Test cases for {@link Ycoord}
 *
 * @author Christian Skärby
 */
public class YcoordTest {

    /**
     * Test that output is correct
     */
    @Test
    public void create() {
        final double[][] test = {{13,15,17},{11, 7, 9}};
        final double[][] expected = {{0,1,2},{0,1,2}};

        final ProcessingResult.Factory proc = new Ycoord();
        final ProcessingResult res = proc.create(new SingletonDoubleInput(test));
        final Optional<double[][]> result = res.stream().findFirst();

        assertTrue("Expected result!", result.isPresent());
        assertArrayEquals("Incorrect output!", expected, result.get());

    }

    /**
     * Test that name is consistent
     */
    @Test
    public void name() {
        assertEquals("Inconsistent name!", Ycoord.nameStatic(), new Ycoord().name());
    }
}