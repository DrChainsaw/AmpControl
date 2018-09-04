package ampcontrol.model.training.model.mutate.reshape;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder(builderClassName = "Builder")
public class ReshapeTask {
    private final long[] targetShape;
    private final long[] sourceShape;
    private final ReshapeSubTask reshapeSubTask;

    public void reshape() {
        reshapeSubTask.execute();
    }
}
