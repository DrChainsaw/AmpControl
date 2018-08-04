package ampcontrol.model.training.schedule.epoch;

import ampcontrol.model.training.schedule.ScheduleBaseTest;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link Offset}
 *
 * @author Christian Skärby
 */
public class OffsetTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new Offset(17, new Linear(2.34));
    }

    @Test
    public void testValueAt() {
        final int offset = 13;
        final ISchedule base = new Linear(3.45);
        final ISchedule sched = new Offset(offset, base);

        assertNotEquals("Need different values for test case to test anything!",
                sched.valueAt(0, 1), sched.valueAt(0, 2));

        IntStream.range(0,7).forEach(epoch ->
        assertEquals("Incorrect value!", base.valueAt(0, offset+epoch), sched.valueAt(0, epoch), 1e-10));
    }

}