package ampcontrol.model.training.schedule;

import ampcontrol.model.training.schedule.epoch.Fixed;
import ampcontrol.model.training.schedule.epoch.Linear;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link Add}
 *
 * @author Christian Skärby
 */
public class AddTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new Add(new Fixed(3.45), new Linear(0.123));
    }

    /**
     * Tests that values are correct
     */
    @Test
    public void testValueAt() {
        final ISchedule first = new Fixed(1.23);
        final ISchedule second = new Linear(3.45);
        final ISchedule sched = new Add(first, second);
        final double expected = first.valueAt(3,5) + second.valueAt(3,5);
        final double actual = sched.valueAt(3,5);
        assertEquals("Incorrect value!", expected, actual, 1e-10);
    }

}