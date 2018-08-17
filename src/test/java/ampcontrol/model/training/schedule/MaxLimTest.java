package ampcontrol.model.training.schedule;

import ampcontrol.model.training.schedule.epoch.Fixed;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link MaxLim}
 *
 * @author Christian Skärby
 */
public class MaxLimTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new MaxLim(2.34, new Fixed(5.67));
    }

    /**
     * Tests that values are correct
     */
    @Test
    public void testValueAt() {
        final double maxVal = 2.34;
        final double less = maxVal / 2;
        final double more = maxVal * 2;
        assertEquals("Incorrect value!", less, new MaxLim(maxVal, new Fixed(less)).valueAt(0,0));
        assertEquals("Incorrect value!", maxVal, new MaxLim(maxVal, new Fixed(more)).valueAt(0,0));
    }
}