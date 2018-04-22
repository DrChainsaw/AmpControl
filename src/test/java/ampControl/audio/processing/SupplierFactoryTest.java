package ampControl.audio.processing;

import org.junit.Test;

import java.util.List;
import java.util.stream.IntStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link SupplierFactory}.
 *
 */
public class SupplierFactoryTest {

    /**
     * Tests that a created {@link ProcessingResult.Factory} can be recreated
     */
    @Test
    public void get() {
        final int sampleRate = 44100;
        final ProcessingResult.Factory pp = new Pipe(
                new Pipe(
                        new Mfsc(sampleRate),
                        new Dct()),
                new Fork(
                        new Pipe(
                                new UnitStdZeroMean(),
                                new Pipe(
                                        new Fork(
                                                new UnitMaxZeroMin(),
                                                new Mfsc(sampleRate)),
                                        new Pipe(
                                                new Spectrogram(4,1),
                                                new Mfsc(sampleRate)
                                        )
                                )
                        ),
                        new Pipe(
                                new UnitMaxZeroMin(),
                                new Mfsc(sampleRate)
                        )));

        final String str = "weewf21_23fd_" + SupplierFactory.prefix() + pp.name() + "_f5re5r7_hy6t8juy45";
        final ProcessingResult.Factory pps = new SupplierFactory(sampleRate).get(str);
        assertEquals("Factory was not restored correctly!", pp.name(), pps.name());

        final ProcessingResult input = new SingletonDoubleInput(IntStream.range(0, 128).mapToDouble(i -> i).toArray());
        final ProcessingResult expected = pp.create(input);
        final ProcessingResult actual = pps.create(input);
        assertEquals("Incorrect number of outputs!", expected.get().size(), actual.get().size());

        final List<double[][]> expectedResult = expected.get();
        final List<double[][]> actualResult = actual.get();
        for(int i = 0; i < expectedResult.size(); i++) {
            assertArrayEquals("Incorrect output nr " + i + "!", expectedResult.get(i),actualResult.get(i));
        }

    }
}