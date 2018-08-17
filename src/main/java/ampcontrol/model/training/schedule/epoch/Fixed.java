package ampcontrol.model.training.schedule.epoch;

import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * "Fixed" schedule. Design is most likely the result of serialization-means-no-refactoring induced insanity...
 *
 * @author Christian Skärby
 */
@Data
public class Fixed implements ISchedule {

    private final double value;

    public Fixed(@JsonProperty("value") double value) {
        this.value = value;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        return value;
    }

    @Override
    public ISchedule clone() {
        return new Fixed(value);
    }
}
