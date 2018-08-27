package ampcontrol.model.training.data.processing;

import org.apache.commons.lang.mutable.MutableDouble;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link WindowedRandomSamplingInfo}
 *
 * @author Christian Skärby
 */
public class WindowedRandomSamplingInfoTest {

    /**
     * Test that the expected sequence of windows is produced.
     */
    @Test
    public void apply() {
        final double[] randomSequence = DoubleStream.of(0, 0.3, 0.77, 1).toArray();
        final int clipLengthMs = 1000;
        final int windowSizeMs = 100;

        final WindowedRandomSamplingInfo infoMap = new WindowedRandomSamplingInfo(clipLengthMs, windowSizeMs, new Random() {
            private int cnt = 0;

            @Override
            public double nextDouble() {
                return randomSequence[cnt++];
            }
        });
        final double[] expectedStart = DoubleStream.of(randomSequence).map(d -> d * (clipLengthMs / 1e3 - windowSizeMs / 1e3)).toArray();
        final double[] actualStart = Stream.generate(() -> Paths.get("asdff"))
                .limit(expectedStart.length)
                .map(infoMap)
                .peek(info -> assertEquals("Incorrect window size!", windowSizeMs / 1e3, info.getLength(), 1e-10))
                .mapToDouble(AudioSamplingInfo::getStartTime).toArray();
        assertArrayEquals("Incorrect start sequence!", expectedStart, actualStart, 1e-10);
    }

    /**
     * Test that and identical sequence is produced if random is reset
     */
    @Test
    public void restoreState() {
        final int clipLengthMs = 1000;
        final int windowSizeMs = 100;
        final double start = 0.111;
        final MutableDouble nextDouble = new MutableDouble(start);
        final WindowedRandomSamplingInfo infoMap = new WindowedRandomSamplingInfo(clipLengthMs, windowSizeMs, new Random() {
            @Override
            public double nextDouble() {
                nextDouble.add(0.23);
                return nextDouble.doubleValue();
            }
        });
        final int nrToDraw = 3;
        final double[] first = Stream.generate(() -> Paths.get("asdff"))
                .limit(nrToDraw)
                .map(infoMap)
                .mapToDouble(AudioSamplingInfo::getStartTime).toArray();

        nextDouble.setValue(start);
        final double[] second = Stream.generate(() -> Paths.get("asdff"))
                .limit(nrToDraw)
                .map(infoMap)
                .mapToDouble(AudioSamplingInfo::getStartTime).toArray();

        assertArrayEquals("Incorrect start sequence!", first, second, 1e-10);
    }
}