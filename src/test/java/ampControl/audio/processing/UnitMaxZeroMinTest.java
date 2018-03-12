package ampControl.audio.processing;

import org.junit.Test;

import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static org.junit.Assert.*;

/**
 * Test cases for {@link UnitMaxZeroMean}.
 *
 * @author Christian Skärby
 */
public class UnitMaxZeroMinTest {

    /**
     * Test that output is normalized
     */
    @Test
    public void receive() {
        final double[][] test = {{1, 2, 3}, {4, 5, 6}};

        final ProcessingResult.Processing norm = new UnitMaxZeroMin();
        norm.receive(test);
        final double[][] result = norm.get().get(0);

        assertEquals("Not unit max!", 1, Stream.of(result).flatMapToDouble(dArr -> DoubleStream.of(dArr)).max().orElseThrow(() -> new RuntimeException("no output!")), 1e-10);
        assertEquals("Not zero min!", 0, Stream.of(result).flatMapToDouble(dArr -> DoubleStream.of(dArr)).min().orElseThrow(() -> new RuntimeException("no output!")), 1e-10);
    }

    /**
     * Test that naming is consistent
     */
    @Test
    public void name() {
        assertEquals("Inconsistent naming!", UnitMaxZeroMin.nameStatic(), new UnitMaxZeroMin().name());
    }
}