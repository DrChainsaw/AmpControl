package ampControl.audio.processing;

import org.junit.Test;

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
        final ProcessingResult.Processing second = new TestProcessing(d -> 2 * d, "");
        final ProcessingResult.Processing pipe = new Pipe(
                new TestProcessing(d -> Math.sqrt(d), ""),
                second);
        pipe.receive(new double[][]{{4, 9, 16}, {25, 36, 49}});
        assertArrayEquals("Incorrect output!", second.get().get(0), pipe.get().get(0));
    }

    /**
     * Test before a {@link Fork}
     */
    @Test
    public void beforeFork() {
        final ProcessingResult.Processing path0 = new TestProcessing(d -> 2 * d, "");
        final ProcessingResult.Processing path1 = new TestProcessing(d -> 3 * d, "");
        final ProcessingResult.Processing pipe = new Pipe(
                new TestProcessing(d -> Math.sqrt(d), ""),
                new Fork(path0, path1));
        pipe.receive(new double[][]{{4, 9, 16}, {25, 36, 49}});
        assertEquals("Incorrect number of outputs!", 2, pipe.get().size());
        assertArrayEquals("Incorrect output!", path0.get().get(0), pipe.get().get(0));
        assertArrayEquals("Incorrect output!", path1.get().get(0), pipe.get().get(1));
    }

    /**
     * Test that name is consistent
     */
    @Test
    public void name() {
        final String nameFirst = "A";
        final String nameSecond = "B";
        final ProcessingResult.Processing pipe = new Pipe(
                new TestProcessing(d -> d, nameFirst),
                new TestProcessing(d -> d, nameSecond));

        assertTrue("First name not present!", pipe.name().contains(nameFirst));
        assertTrue("Second name not present!", pipe.name().contains(nameSecond));
        assertEquals("Inconsistent name!", Pipe.nameStatic(),
                pipe.name().replace(nameFirst, "").replace(nameSecond, ""));

    }

}