package ampcontrol.model.training.schedule;

import ampcontrol.model.training.schedule.epoch.Fixed;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import static junit.framework.TestCase.assertEquals;

public class MinLimTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new MinLim(2.34, new Fixed(5.67));
    }

    /**
     * Tests that values are correct
     */
    @Test
    public void testValueAt() {
        final double minVal = 2.34;
        final double less = minVal / 2;
        final double more = minVal * 2;
        assertEquals("Incorrect value!", minVal, new MinLim(minVal, new Fixed(less)).valueAt(0,0));
        assertEquals("Incorrect value!", more, new MinLim(minVal, new Fixed(more)).valueAt(0,0));
    }
}