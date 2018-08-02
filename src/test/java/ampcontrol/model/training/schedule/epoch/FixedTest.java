package ampcontrol.model.training.schedule.epoch;

import ampcontrol.model.training.schedule.ScheduleBaseTest;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link Fixed}
 *
 * @author Christian Skärby
 */
public class FixedTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new Fixed(666);
    }

    /**
     * Tests that values are correct
     */
    @Test
    public void testValueAt() {
        final double val = 0.1234;
        final ISchedule sched = new Fixed(val);
        assertEquals("Incorrect value!", val, sched.valueAt(123,456), 1e-10);
    }
}