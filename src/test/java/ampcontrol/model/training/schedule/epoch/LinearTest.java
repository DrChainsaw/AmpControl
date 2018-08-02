package ampcontrol.model.training.schedule.epoch;

import ampcontrol.model.training.schedule.ScheduleBaseTest;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link Linear}
 *
 * @author Christian Skärby
 */
public class LinearTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new Linear(6.66);
    }

    /**
     * Test values produced
     */
    @Test
    public void testValueAt() {
        final double factor = 13.3;
        final ISchedule sched = new Linear(factor);
        IntStream.range(0,7).forEach(epoch ->
                assertEquals("Incorrect value!",epoch * factor, sched.valueAt(666, epoch), 1e-10));
    }
}