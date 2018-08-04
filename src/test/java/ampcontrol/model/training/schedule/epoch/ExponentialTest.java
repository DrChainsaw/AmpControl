package ampcontrol.model.training.schedule.epoch;

import ampcontrol.model.training.schedule.ScheduleBaseTest;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link Exponential}.
 *
 * @author Christian Skärby
 */
public class ExponentialTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new Exponential(6.66);
    }

    /**
     * Test values produced
     */
    @Test
    public void testValueAt() {
        final double base = 3;
        final ISchedule sched = new Exponential(base);
        assertEquals("Incorrect value!", 1, sched.valueAt(666, 0), 1e-10);
        assertEquals("Incorrect value!", base, sched.valueAt(666, 1), 1e-10);
        assertEquals("Incorrect value!", base*base, sched.valueAt(666, 2), 1e-10);
        assertEquals("Incorrect value!", base*base*base, sched.valueAt(666, 3), 1e-10);
    }
}