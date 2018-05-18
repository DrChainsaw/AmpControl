package ampcontrol.audio.processing;

import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

/**
 * Test cases for {@link Pipe}.
 *
 * @author Christian Skärby
 */
public class PipeTest {

    /**
     * Test that output is passed through correctly
     */
    @Test
    public void receive() {
        final ProcessingResult.Factory pipe = new Pipe(
                new TestProcessing(d -> Math.sqrt(d), ""),
                new TestProcessing(d -> 2 * d, ""));
        final ProcessingResult input = new SingletonDoubleInput(new double[][]{{4, 9, 16}, {25, 36, 49}});
        final double[][] expected = new double[][]{{4, 6, 8}, {10, 12, 14}};
        final ProcessingResult res = pipe.create(input);
        assertArrayEquals("Incorrect output!", expected, res.stream().findFirst().get());
    }

    /**
     * Test before a {@link Fork}
     */
    @Test
    public void beforeFork() {
        final ProcessingResult.Factory path0 = new TestProcessing(d -> 2 * d, "");
        final ProcessingResult.Factory path1 = new TestProcessing(d -> 3 * d, "");
        final ProcessingResult.Factory pipe = new Pipe(
                new TestProcessing(d -> Math.sqrt(d), ""),
                new Fork(path0, path1));
        final ProcessingResult res = pipe.create(new SingletonDoubleInput(new double[][]{{4, 9, 16}, {25, 36, 49}}));
        final List<double[][]> resList = res.stream().collect(Collectors.toList());
        final double[][] expected0 = new double[][]{{4, 6, 8}, {10, 12, 14}};
        final double[][] expected1 = new double[][]{{6, 9, 12}, {15, 18, 21}};
        assertEquals("Incorrect number of outputs!", 2, resList.size());
        assertArrayEquals("Incorrect output!", expected0, resList.get(0));
        assertArrayEquals("Incorrect output!", expected1, resList.get(1));
    }

    /**
     * Test that name is consistent
     */
    @Test
    public void name() {
        final String nameFirst = "A";
        final String nameSecond = "B";
        final ProcessingResult.Factory pipe = new Pipe(
                new TestProcessing(d -> d, nameFirst),
                new TestProcessing(d -> d, nameSecond));

        assertTrue("First name not present!", pipe.name().contains(nameFirst));
        assertTrue("Second name not present!", pipe.name().contains(nameSecond));
        assertEquals("Inconsistent name!", Pipe.nameStatic(),
                pipe.name().replace(nameFirst, "").replace(nameSecond, ""));

    }

}