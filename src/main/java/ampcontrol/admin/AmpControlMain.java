package ampcontrol.admin;

import ampcontrol.admin.service.classifiction.AudioClassificationService;
import ampcontrol.admin.service.control.mqtt.MqttAppControlService;
import ampcontrol.amp.AmpInterface;
import ampcontrol.amp.ClassificationListener;
import ampcontrol.amp.PublishingClassificationListener;
import ampcontrol.audio.asio.AsioClassifierInputFactory;
import ampcontrol.model.inference.Classifier;
import ampcontrol.model.inference.ClassifierFromParameters;
import ampcontrol.model.training.model.vertex.ChannelMultVertex;
import ampcontrol.model.training.model.vertex.ElementWiseVertexLatest;
import com.beust.jcommander.JCommander;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Map;

/**
 * Main class for doing AmpControl. Just does initialization. Actual main loop is handled by the {@link Engine}.
 *
 * @author Christian Skärby
 */
public class AmpControlMain {

    public  static void main(String[] args) {

        // Class not under test. Is it even testable in practice?
        final Engine engine = new Engine();
        final MqttAppControlService mqttAppControlService = new MqttAppControlService();
        final PublishingClassificationListener.Factory mqttClassificationListenerFactory =
                new PublishingClassificationListener.Factory(mqttAppControlService);
        final AudioClassificationService audioClassificationService = new AudioClassificationService();
        final ClassifierFromParameters classifierFromParameters = new ClassifierFromParameters();
        final AsioClassifierInputFactory inputProviderFactory = new AsioClassifierInputFactory();

        JCommander.Builder jcBuilder = JCommander.newBuilder()
                .addObject(new Object[] {
                        engine,
                        mqttAppControlService,
                        audioClassificationService,
                        inputProviderFactory,
                        classifierFromParameters,
                        mqttClassificationListenerFactory});

        Map<String, AmpInterface.Factory> ampFactoryCommands = AmpInterface.getFactoryCommands();
        ampFactoryCommands.forEach((key, value) -> jcBuilder.addCommand(key, value));

        JCommander jc = jcBuilder.build();
        jc.parse(args);

        final AmpInterface ampInterface = ampFactoryCommands.get(jc.getParsedCommand()).create();
        final ClassificationListener mqttInterface = mqttClassificationListenerFactory.create();
        final ClassificationListener classificationListenerAgg = arr -> {
        	ampInterface.indicateAudioClassification(arr);
            mqttInterface.indicateAudioClassification(arr);
        };
        inputProviderFactory.initialize();

        try {
            // Might need to move into concrete Classifiers if something else is used in training
            DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
            Nd4j.getMemoryManager().setAutoGcWindow(5000);
            NeuralNetConfiguration.registerLegacyCustomClassesForJSON(ChannelMultVertex.class, ElementWiseVertexLatest.class);
            final Classifier classifier = classifierFromParameters.getClassifier(inputProviderFactory);
            audioClassificationService.initialize(
                    classificationListenerAgg,
                    classifier,
                    inputProviderFactory.finalizeAndReturnUpdateHandle());

            engine.initialize(mqttAppControlService, Arrays.asList(
                    audioClassificationService,
                    ampInterface));
            engine.run();
        } catch (Exception e) {
            System.out.println("Application failed!");
            e.printStackTrace();
            System.exit(1);
        }

    }
}
