package ampControl.audio.processing;

import org.junit.Test;

import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link UnitStdZeroMean}.
 *
 * @author Christian Skärby
 */
public class UnitStdZeroMeanTest {

    /**
     * Test that output is standardized
     */
    @Test
    public void receive() {
        final double[][] test = {{1, 2, 3}, {4, 5, 6}};
        final int nrofSamples = 6;

        final ProcessingResult.Factory uszm = new UnitStdZeroMean();
        final ProcessingResult res = uszm.create(new SingletonDoubleInput(test));
        final double[][] result = res.get().get(0);

        assertEquals("Not unit std!" , 1,Stream.of(result).flatMapToDouble(dArr -> DoubleStream.of(dArr)).map(d -> d*d).sum() / nrofSamples, 1e-10);
        assertEquals("Not zero mean!", 0, Stream.of(result).flatMapToDouble(dArr -> DoubleStream.of(dArr)).sum(), 1e-10);
    }

    /**
     * Test that naming is consistent
     */
    @Test
    public void name() {
        assertEquals("Inconsistent naming!", UnitStdZeroMean.nameStatic(), new UnitStdZeroMean().name());
    }

}