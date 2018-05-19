package ampcontrol.amp;

import ampcontrol.admin.service.control.TopicPublisher;
import ampcontrol.amp.labelmapping.*;
import ampcontrol.amp.probabilities.ArgMax;
import ampcontrol.amp.probabilities.Interpreter;
import ampcontrol.amp.probabilities.ThresholdFilter;
import com.beust.jcommander.Parameter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;

/**
 * Publishes classifications to a {@link TopicPublisher}
 *
 * @author Christian Skärby
 */
public class PublishingClassificationListener implements ClassificationListener {

    private static final Logger log = LoggerFactory.getLogger(PublishingClassificationListener.class);

    private final String classificationTopic;
    private final TopicPublisher topicPublisher;
    private final Function<INDArray, List<Integer>> labelMapper;

    public static class Factory implements ClassificationListener.Factory {

        private final TopicPublisher topicPublisher;

        @Parameter(names = "-mqttClassificationTopic", description = "MQTT topic to publish labels to")
        private String classificationTopic = "podxtcontrol/classification/label";

        @Parameter(names = {"-probThreshToReport", "-pth"}, description = "Probability must be higher than this to publish result")
        private double probThresh = 0.8;

        @Parameter(names = {"-momentumThreshold", "-mth"}, description = "Must get at least this many consecutive classification to publish result")
        private int momentumThresh = 10;

        @Parameter(names = {"-updateProhibitTime", "-upt"}, description = "Shortest time in milli seconds between new updates")
        private int updateProhibitTimeMs = 1000;

        public Factory(TopicPublisher topicPublisher) {
            this.topicPublisher = topicPublisher;
        }

        @Override
        public ClassificationListener create() {
            return new PublishingClassificationListener(classificationTopic, topicPublisher, createClassificationFilters());
        }

        private Function<INDArray, List<Integer>> createClassificationFilters() {
            Interpreter<Integer> interpretProbabilities = new ArgMax();
            for (int i = 0; i < 4; i++) {
                interpretProbabilities = new ThresholdFilter<>(i, probThresh, interpretProbabilities);
            }

            final LabelMapping<Integer> mapLabelToProgram =
                    new MomentumLabelMapping<>(momentumThresh,
                            new MaskDuplicateLabelMapping<>(
                                    new PacingLabelMapping<>(updateProhibitTimeMs,
                                            Collections::singletonList
                                    )
                            )
                    );
            final int invalidLabel = 6666;
            return interpretProbabilities
                    .andThen(list -> {
                        if (!list.isEmpty()) {
                            return list.get(0);
                        }
                        return invalidLabel;
                    })
                    .andThen(new MaskingLabelMapping<>(Collections.singletonList(invalidLabel), mapLabelToProgram));
        }
    }

    private PublishingClassificationListener(
            String classificationTopic,
            TopicPublisher topicPublisher,
            Function<INDArray, List<Integer>> labelMapper) {
        this.classificationTopic = classificationTopic;
        this.topicPublisher = topicPublisher;
        this.labelMapper = labelMapper;
    }

    @Override
    public void indicateAudioClassification(INDArray probabilities) {
        labelMapper.apply(probabilities).stream().findAny()
                .ifPresent(label -> {
                    final String labelStr = ""+label;
                        log.info("Publish: " + labelStr);
                        topicPublisher.publish(classificationTopic, labelStr);
                });
    }

}
