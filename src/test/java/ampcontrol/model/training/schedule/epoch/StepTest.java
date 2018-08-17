package ampcontrol.model.training.schedule.epoch;

import ampcontrol.model.training.schedule.ScheduleBaseTest;
import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;

import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link Step}
 *
 * @author Christian Skärby
 */
public class StepTest extends ScheduleBaseTest {

    @Override
    protected ISchedule createBaseTestInstance() {
        return new Step(17, new Linear(11));
    }

    /**
     * Tests that values are correct
     */
    @Test
    public void testValueAt() {
        final int step = 11;
        final ISchedule sched = new Step(step, new Linear(17));

        assertNotEquals("Values not changing!",
                sched.valueAt(0, 1), sched.valueAt(0, 1+step));

        IntStream.range(0, step).forEach(epoch ->
                assertEquals("Incorrect value at epoch " +epoch + "!",
                        sched.valueAt(1, epoch),
                        sched.valueAt(13, epoch - 1), 1e-10));
    }

}