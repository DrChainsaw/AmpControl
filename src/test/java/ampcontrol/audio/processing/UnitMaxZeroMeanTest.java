package ampcontrol.audio.processing;

import org.junit.Test;

import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link UnitMaxZeroMeanTest}
 */
public class UnitMaxZeroMeanTest {

    /**
     * Test that output is normalized
     */
    @Test
    public void receive() {
        final double[][] test = {{1, 2, 3}, {4, 5, 6}};

        final ProcessingResult.Factory umzm = new UnitMaxZeroMean();
        final ProcessingResult res = umzm.create(new SingletonDoubleInput(test));
        final double[][] result = res.stream().findFirst().get();

        assertEquals("Not unit max!", 1, Stream.of(result).flatMapToDouble(DoubleStream::of).max().orElseThrow(() -> new RuntimeException("no output!")), 1e-10);
        assertEquals("Not zero mean!", 0, Stream.of(result).flatMapToDouble(DoubleStream::of).sum(), 1e-10);
    }

    /**
     * Test that naming is consistent
     */
    @Test
    public void name() {
        assertEquals("Inconsistent naming!", UnitMaxZeroMean.nameStatic(), new UnitMaxZeroMean().name());
    }

}