package ampcontrol.audio.processing;

import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ProcessingFactoryFromString}.
 *
 */
public class ProcessingFactoryFromStringTest {

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
                                                new ZeroMean()),
                                        new Pipe(
                                                new Spectrogram(4,1),
                                                new Log10()
                                        )
                                )
                        ),
                        new Pipe(
                                new NoProcessing(),
                                new LogScale()
                        )));

        final String str = "weewf21_23fd_" + ProcessingFactoryFromString.prefix() + pp.name() + "_f5re5r7_hy6t8juy45";
        final ProcessingResult.Factory pps = new ProcessingFactoryFromString(sampleRate).get(str);
        assertEquals("Factory was not restored correctly!", pp.name(), pps.name());

        final ProcessingResult input = new SingletonDoubleInput(IntStream.range(0, 128).mapToDouble(i -> i).toArray());
        final ProcessingResult expected = pp.create(input);
        final ProcessingResult actual = pps.create(input);
        final List<double[][]> expectedResult = expected.stream().collect(Collectors.toList());
        final List<double[][]> actualResult = actual.stream().collect(Collectors.toList());

        assertEquals("Incorrect number of outputs!", expectedResult.size(), actualResult.size());
        for(int i = 0; i < expectedResult.size(); i++) {
            assertArrayEquals("Incorrect output nr " + i + "!", expectedResult.get(i),actualResult.get(i));
        }

    }
}