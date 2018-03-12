package ampControl.audio.processing;

import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.*;

/**
 * Test case for {@link Fork}.
 *
 * @author Christian Skärby
 */
public class ForkTest {

    /**
     * Test output from a single fork
     */
    @Test
    public void receiveSingle() {
        final ProcessingResult.Processing mock0 = createMock("A", 2);
        final ProcessingResult.Processing mock1 = createMock("B", 666);
        final ProcessingResult.Processing fork = new Fork(mock0, mock1);
        fork.receive(new double[][]{{0.1, 0.2}, {0.3, 0.4}});

        assertEquals("Incorrect size!", 2, fork.get().size());

        assertArrayEquals("Incorrect output path 1!", mock0.get().get(0), fork.get().get(0));
        assertArrayEquals("Incorrect output path 2!", mock1.get().get(0), fork.get().get(1));
    }

    /**
     * Test output from a multi-fork
     */
    @Test
    public void receiveMulti() {
        List<ProcessingResult.Processing> mocks = IntStream.range(0, 10).mapToObj(i -> createMock("mock" + i, i)).collect(Collectors.toList());
        ProcessingResult.Processing last = null;
        for (ProcessingResult.Processing mock : mocks) {
            if (last == null) {
                last = mock;
            } else {
                last = new Fork(last, mock);
            }
        }
        last.receive(new double[][]{{0.1, 0.2}, {0.3, 0.4}});

        assertEquals("Incorrect size!", mocks.size(), last.get().size());
        for (int i = 0; i < mocks.size(); i++) {
            assertArrayEquals("Incorrect output path "+ i +"!", mocks.get(i).get().get(0), last.get().get(i));
        }
    }

    /**
     * Test before a {@link Pipe}
     */
    @Test
    public void withPipe() {
        final ProcessingResult.Processing path0 = new TestProcessing(d -> 2 * d, "");
        final ProcessingResult.Processing path1 = new TestProcessing(d -> 3 * d, "");
        final ProcessingResult.Processing fork = new Fork(
                new Pipe(new TestProcessing(d -> Math.sqrt(d), ""), path0),
                path1
        );
        fork.receive(new double[][]{{4, 9, 16}, {25, 36, 49}});
        assertEquals("Incorrect number of outputs!", 2, fork.get().size());
        assertArrayEquals("Incorrect output!", path0.get().get(0), fork.get().get(0));
        assertArrayEquals("Incorrect output!", path1.get().get(0), fork.get().get(1));
    }

    /**
     * Test that name can be reversed into components
     *
     */
    @Test
    public void name() {
        final ProcessingResult.Processing mock0 = createMock("A", 2);
        final ProcessingResult.Processing mock1 = createMock("B", 666);
        final ProcessingResult.Processing fork = new Fork(mock0, mock1);

        assertTrue("Name mismatch!", fork.name().matches(Fork.matchStrStatic()));
        final String first = Fork.splitFirst(fork.name())[1];
        final String[] next = Fork.splitMid(first);
        assertEquals("Incorrect name!", mock0.name(), next[0]);
        assertEquals("Incorrect name!", mock1.name(), Fork.splitEnd(next[1])[0]);
    }


    private static ProcessingResult.Processing createMock(final String name, final double resultScale) {
        return new TestProcessing(d -> d*resultScale, name);
    }

}