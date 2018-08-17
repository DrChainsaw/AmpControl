package ampcontrol.model.training.schedule.epoch;

import ampcontrol.model.training.schedule.ScheduleBaseTest;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link Cyclic}
 *
 * @author Christian Skärby
 */
public class CyclicTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new Cyclic(13, new Linear(17));
    }

    /**
     * Tests that values are correct
     */
    @Test
    public void testValueAt() {
        final int period = 11;
        final ISchedule sched = new Cyclic(period, new Linear(23));

        assertNotEquals("Need different values for test case to test anything!",
                sched.valueAt(0, 1), sched.valueAt(0, 2));

        IntStream.range(0, 2*period).forEach(epoch ->
                assertEquals("Incorrect value at epoch " +epoch + "!",
                        sched.valueAt(1, epoch),
                        sched.valueAt(13, epoch + period), 1e-10));
    }

}