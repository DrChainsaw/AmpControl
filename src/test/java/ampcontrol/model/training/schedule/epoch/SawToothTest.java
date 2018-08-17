package ampcontrol.model.training.schedule.epoch;

import ampcontrol.model.training.schedule.ScheduleBaseTest;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.DoubleSummaryStatistics;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link SawTooth}
 *
 * @author Christian Skärby
 */
public class SawToothTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new SawTooth(666,0.123,4.567);
    }

    /**
     * Tests that values are correct
     */
    @Test
    public void testValueAtEpoch() {
        final int period = 100;
        final double minLr = 1e-5;
        final double maxLr = 1e-2;

        final ISchedule sched = new SawTooth(period, minLr, maxLr);

        List<Double> twoPeriods = IntStream.range(0, 2 * period)
                .mapToDouble(epoch -> sched.valueAt(0, epoch))
                .boxed()
                .collect(Collectors.toList());
        DoubleSummaryStatistics stats = twoPeriods.stream().mapToDouble(d -> d).summaryStatistics();

        assertEquals("Incorrect max learning rate!", maxLr, stats.getMax(), 1e-10);
        assertEquals("Incorrect min learning rate!", minLr, stats.getMin(), 1e-10);
        assertEquals("Incorrect avg learning rate!", (minLr + maxLr) / 2, stats.getAverage(), 1e-10);

        final Set<Double> diffs = new HashSet<>();
        twoPeriods.stream().mapToDouble(d -> d)
                .reduce((d1, d2) -> {
                    diffs.add((double)Math.round(1e6*(d1 - d2)) / 1e6);
                    return d2;
                });
        assertEquals("Incorrect diffs!", 2, diffs.size());
        assertEquals("Incorrect diffs!", 0.0, diffs.stream().mapToDouble(d -> d).sum(), 1e-10);
    }
}