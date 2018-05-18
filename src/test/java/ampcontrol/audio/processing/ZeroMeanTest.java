package ampcontrol.audio.processing;

import org.junit.Test;

import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ZeroMean}
 *
 * @author Christian Skärby
 */
public class ZeroMeanTest {

    /**
     * Test that ouput is zero mean
     */
    @Test
    public void receive() {
        final double[][] test = {{1, 2, 3}, {4, 5, 6}};

        final ProcessingResult.Factory proc = new UnitMaxZeroMean();
        final ProcessingResult res = proc.create(new SingletonDoubleInput(test));
        final double[][] result = res.stream().findFirst().get();

        assertEquals("Not zero mean!", 0, Stream.of(result).flatMapToDouble(DoubleStream::of).sum(), 1e-10);
    }

    /**
     * Test that naming is consistent
     */
    @Test
    public void name() {
        assertEquals("Inconsistent naming!", ZeroMean.nameStatic(), new ZeroMean().name());
    }
}