package ampcontrol.model.training.schedule;

import org.junit.Test;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;

import static junit.framework.TestCase.fail;
import static org.junit.Assert.*;

/**
 * Tests functionality common to all {@link ISchedule}s, such as clone and serialization/deserialization
 *
 * @author Christian Skärby
 */
public abstract class ScheduleBaseTest {

    protected abstract ISchedule createBaseTestInstance();

    /**
     * Test cloning
     */
    @Test
    public void testClone() {
        final ISchedule sched = createBaseTestInstance();
        final ISchedule clone = sched.clone();
        assertNotSame("Different instance expected!", sched, clone);
        assertEquals("Expected instances to be equal!", sched, clone);
        assertEquals("Incorrect value!", sched.valueAt(123,456), clone.valueAt(123,456), 1e-10);
    }

    /**
     * Test JSON serialization
     */
    @Test
    public void testJson() {
        final ISchedule sched = createBaseTestInstance();
        final ObjectMapper mapper = new ObjectMapper();
        try {
            final String json = mapper.writeValueAsString(sched);
            final ISchedule deserialized = mapper.readValue(json, sched.getClass());
            assertEquals("Incorrect value!", sched.valueAt(123,456), deserialized.valueAt(123,456), 1e-10);
        } catch (IOException e) {
            fail(e.getMessage());
        }
    }

    /**
     * Test equals
     */
    @Test
    public void testEquals() {
        final ISchedule sched = createBaseTestInstance();
        assertNotEquals("Shall not be equal to a generic object!", sched, new Object());
        assertEquals("Shall be equal to identical instance!", sched, createBaseTestInstance());
    }

    /**
     * Test hash code
     */
    @Test
    public void testHashCode() {
        final ISchedule sched = createBaseTestInstance();
        assertNotEquals("Shall not be equal to a generic object!", sched.hashCode(), new Object().hashCode());
        assertEquals("Shall be equal to identical instance!", sched.hashCode(), createBaseTestInstance().hashCode());
    }
}
