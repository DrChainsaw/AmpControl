package ampcontrol.audio.processing;

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
        final ProcessingResult.Factory mock0 = createMock("A", 2);
        final ProcessingResult.Factory mock1 = createMock("B", 666);
        final ProcessingResult.Factory fork = new Fork(mock0, mock1);
        final ProcessingResult input = new SingletonDoubleInput(new double[][] {{0.1, 0.2}, {0.3, 0.4}});
        final ProcessingResult res = fork.create(input);
        final List<double[][]> resList = res.stream().collect(Collectors.toList());

        assertEquals("Incorrect size!", 2, resList.size());
        assertArrayEquals("Incorrect output path 1!", mock0.create(input).stream().findFirst().get(), resList.get(0));
        assertArrayEquals("Incorrect output path 2!", mock1.create(input).stream().findFirst().get(), resList.get(1));
    }

    /**
     * Test output from a multi-fork
     */
    @Test
    public void receiveMulti() {
        List<ProcessingResult.Factory> mocks = IntStream.range(0, 10).mapToObj(i -> createMock("mock" + i, i)).collect(Collectors.toList());
        ProcessingResult.Factory last = null;
        for (ProcessingResult.Factory mock : mocks) {
            if (last == null) {
                last = mock;
            } else {
                last = new Fork(last, mock);
            }
        }
        final ProcessingResult input = new SingletonDoubleInput(new double[][]{{0.1, 0.2}, {0.3, 0.4}});
        final ProcessingResult res = last.create(input);
        final List<double[][]> resList = res.stream().collect(Collectors.toList());

        assertEquals("Incorrect size!", mocks.size(), resList.size());
        for (int i = 0; i < mocks.size(); i++) {
            assertArrayEquals("Incorrect output path "+ i +"!", mocks.get(i).create(input).stream().findFirst().get(), resList.get(i));
        }
    }

    /**
     * Test before a {@link Pipe}
     */
    @Test
    public void withPipe() {
        final ProcessingResult.Factory path0 = new Pipe(
                new TestProcessing(d -> Math.sqrt(d), ""),
                new TestProcessing(d -> 2 * d, ""));
        final ProcessingResult.Factory path1 = new TestProcessing(d -> 3 * d, "");
        final ProcessingResult.Factory fork = new Fork(
                path0,
                path1
        );
        final ProcessingResult input = new SingletonDoubleInput(new double[][]{{4, 9, 16}, {25, 36, 49}});
        final ProcessingResult res = fork.create(input);
        final List<double[][]> resList = res.stream().collect(Collectors.toList());

        assertEquals("Incorrect number of outputs!", 2, resList.size());
        assertArrayEquals("Incorrect output!", path0.create(input).stream().findFirst().get(), resList.get(0));
        assertArrayEquals("Incorrect output!", path1.create(input).stream().findFirst().get(), resList.get(1));
    }

    /**
     * Test that name can be reversed into components
     *
     */
    @Test
    public void name() {
        final ProcessingResult.Factory mock0 = createMock("A", 2);
        final ProcessingResult.Factory mock1 = createMock("B", 666);
        final ProcessingResult.Factory fork = new Fork(mock0, mock1);

        assertTrue("Name mismatch!", fork.name().matches(Fork.matchStrStatic()));
        final String first = Fork.splitFirst(fork.name())[1];
        final String[] next = Fork.splitMid(first);
        assertEquals("Incorrect name!", mock0.name(), next[0]);
        assertEquals("Incorrect name!", mock1.name(), Fork.splitEnd(next[1])[0]);
    }


    private static ProcessingResult.Factory createMock(final String name, final double resultScale) {
        return new TestProcessing(d -> d*resultScale, name);
    }

}