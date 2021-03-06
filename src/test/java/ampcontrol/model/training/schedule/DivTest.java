package ampcontrol.model.training.schedule;

import ampcontrol.model.training.schedule.epoch.Fixed;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link Div}
 *
 * @author Christian Skärby
 */
public class DivTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new Div(new Fixed(21.3), new Fixed(1.77));
    }

    /**
     * Tests that values are correct
     */
    @Test
    public void testValueAt() {
        final ISchedule first = new Fixed(1.23);
        final ISchedule second = new Fixed(2.34);
        final ISchedule sched = new Div(first, second);
        final double expected = first.valueAt(1,2) / second.valueAt(1,2);
        final double actual = sched.valueAt(1,2);
        assertEquals("Incorrect value", expected, actual, 1e-10);
    }

}