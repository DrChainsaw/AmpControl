package ampcontrol.model.training.data.processing;

import ampcontrol.model.training.data.state.ResetableStateFactory;
import ampcontrol.model.training.data.state.SimpleStateFactory;
import org.junit.Test;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link WindowedConsecutiveSamplingInfo}
 *
 * @author Christian Skärby
 */
public class WindowedConsecutiveSamplingInfoTest {

    /**
     * Test that consecutive windows are produced when clip length is an integer multiple of window size
     */
    @Test
    public void applyEvenSize() {
        final int clipLengthMs = 1000;
        final int windowSizeMs = 100;
        testApply(clipLengthMs, windowSizeMs);
    }

    /**
     * Test that consecutive windows are produced when clip length is not an integer multiple of window size
     */
    @Test
    public void applyOddSize() {
        final int clipLengthMs = 93;
        final int windowSizeMs = 7;
        testApply(clipLengthMs, windowSizeMs);
    }

    private static void testApply(int clipLengthMs, int windowSizeMs) {
        final Map<Path, Double> pathToExpectedStart = Stream.of(
                Paths.get("iigdkds"),
                Paths.get("dslkgr")
        ).collect(Collectors.toMap(Function.identity(), path -> 0d));

        final WindowedConsecutiveSamplingInfo infoMap = new WindowedConsecutiveSamplingInfo(clipLengthMs, windowSizeMs, new SimpleStateFactory(666));
        for (int i = 0; i < 2 * clipLengthMs / windowSizeMs; i++) {
            for (Map.Entry<Path, Double> pathStartEntry : pathToExpectedStart.entrySet()) {
                final AudioSamplingInfo info = infoMap.apply(pathStartEntry.getKey());
                assertEquals("Incorrect start!", pathStartEntry.getValue(), info.getStartTime(), 1e-10);
                assertEquals("Incorrect length!", windowSizeMs / 1e3, info.getLength(), 1e-10);
                final double nextStart = info.getStartTime() + info.getLength();
                if (nextStart > clipLengthMs / 1e3 - windowSizeMs / 1e3) {
                    pathStartEntry.setValue(0d);
                } else {
                    pathStartEntry.setValue(nextStart);
                }
            }
        }
    }

    /**
     * Test that an identical sequence is produced after state has been reset
     */
    @Test
    public void restoreState() {
        final ResetableStateFactory stateFactory = new ResetableStateFactory(666);
        final WindowedConsecutiveSamplingInfo testInfo = new WindowedConsecutiveSamplingInfo(1000, 100, stateFactory);
        final List<Double> first = drawSamplingInfoFrom(testInfo);
        stateFactory.storeCurrentState();

        final List<Double> second = drawSamplingInfoFrom(testInfo);
        assertNotEquals("Expected different outputs!", first, second);
        stateFactory.restorePreviousState();

        final List<Double> third = drawSamplingInfoFrom(testInfo);
        assertEquals("Expected same output!", second, third);
    }

    private static List<Double> drawSamplingInfoFrom(WindowedConsecutiveSamplingInfo testInfo) {
        // Draw 13 samples from each "file" (named 0,1,2,3,4) in interleaved order
        return IntStream.range(0, 13).boxed().flatMap(i ->
                IntStream.range(0, 5)
                        .mapToObj(String::valueOf)
                        .map(Paths::get)
                        .map(testInfo)
                        .map(AudioSamplingInfo::getStartTime))
                .collect(Collectors.toList());
    }

}