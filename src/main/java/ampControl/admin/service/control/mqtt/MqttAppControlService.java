package ampControl.admin.service.control.mqtt;

import ampControl.admin.service.control.AppControlService;
import ampControl.admin.service.control.MessageToActionMap;
import ampControl.admin.service.control.TopicPublisher;
import org.eclipse.paho.client.mqttv3.*;

import com.beust.jcommander.Parameter;

/**
 * Service for interacting with the application through MQTT.
 *
 * @author Christian Skärby
 */
public class MqttAppControlService implements TopicPublisher, AppControlService {

    @Parameter(names = "-mqttServer", description = "MQTT server to connect to")
    private String server = "";

    @Parameter(names = "-mqttUsername", description = "MQTT server username. Leave empty if none")
    private String uname = "";

    @Parameter(names = "-mqttPwd", description = "MQTT server password. Leave empty if none")
    private String pwd = "";

    @Parameter(names = "-mqttListenTopic", description = "MQTT topic to subscribe to")
    private String listenTopic = "podxtcontrol/commands";

    @Parameter(names = "-mqttOnlineTopic", description = "MQTT topic to publish to when service goes online/offline")
    private String onlinePublishTopic = "podxtcontrol/online";

    private final static MqttMessage ON = new MqttMessage("ON".getBytes());
    private final static MqttMessage OFF = new MqttMessage("OFF".getBytes());

    private final MqttCallbackMap callbackMapper = new MqttCallbackMap();
    private ClientFactory clientFactory = serverUri -> new MqttClient(serverUri, MqttClient.generateClientId());
    private IMqttClient client;

    /**
     * Interface for client creation. Intended for testing
     */
    interface ClientFactory {
        /**
         * Create a new {@link IMqttClient} to the given serverUri
         *
         * @param serverUri
         * @return
         * @throws MqttException
         */
        IMqttClient create(String serverUri) throws MqttException;
    }

    /**
     * Starts the service by connecting to the broker. Returns a {@link MessageToActionMap} for users to define
     * actions in response to messages on the listenTopic.
     *
     * @return Returns a {@link MessageToActionMap}
     * @throws MqttException
     */
    @Override
    public MessageToActionMap start() throws MqttException {

        client = clientFactory.create(server); //new MqttClient(server, MqttClient.generateClientId());

        MqttConnectOptions connectOptions = new MqttConnectOptions();
        connectOptions.setCleanSession(true);

        connectOptions.setUserName(uname);
        connectOptions.setPassword(pwd.toCharArray());
        connectOptions.setWill(onlinePublishTopic, OFF.getPayload(), 0, false);

        client.connect(connectOptions);
        client.publish(onlinePublishTopic, ON);

        client.setCallback(callbackMapper);
        client.subscribe(listenTopic);
        return callbackMapper;
    }

    /**
     * Disconnects from the broker.
     *
     * @throws MqttException
     */
    @Override
    public void stop() throws MqttException {
    	client.publish(onlinePublishTopic, OFF);
    	client.disconnect();
    }

    @Override
    public void publish(String topic, String message) {
        try {
        client.publish(topic, new MqttMessage(message.getBytes()));
        } catch (MqttException e) {
            System.err.println("Classification message failed!");
            e.printStackTrace();
        }
    }

    /**
     * Sets the {@link ClientFactory} to use. Intended for testing
     *
     * @param clientFactory
     */
    void setClientFactory(ClientFactory clientFactory) {
        this.clientFactory = clientFactory;
    }
}
