package ampcontrol.model.training.schedule.epoch;

import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Offsets the epoch by a fixed amount. Design is most likely the result of serialization-means-no-refactoring induced
 * insanity...
 *
 * @author Christian Skärby
 */
@Data
public class Offset implements ISchedule {

    private final int offset;
    private final ISchedule base;

    public Offset(
            @JsonProperty("offset") int offset,
            @JsonProperty("base") ISchedule base) {
        this.offset = offset;
        this.base = base;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        return base.valueAt(iteration, epoch+offset);
    }

    @Override
    public ISchedule clone() {
        return new Offset(offset, base.clone());
    }
}
