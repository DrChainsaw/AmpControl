package ampControl.admin.service.control.mqtt;

import org.eclipse.paho.client.mqttv3.*;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Test mock for {@link IMqttClient}. Probably madness...
 */
public class MockMqttClient implements IMqttClient {

    private final Set<String> subscribedTopics = new HashSet<>();
    private String lastPublishedTopic = "";
    private String lastPublishedMessage = "";

    private boolean isConnected = false;

    public Set<String> getSubscribedTopics() {
        return subscribedTopics;
    }

    public String getLastPublishedTopic() {
        return lastPublishedTopic;
    }

    public String getLastPublishedMessage() {
        return lastPublishedMessage;
    }

    public void clearLastPublished() {
        lastPublishedMessage = "";
        lastPublishedTopic = "";
    }

    @Override
    public void connect() throws MqttSecurityException, MqttException {
        isConnected = true;
    }

    @Override
    public void connect(MqttConnectOptions options) throws MqttSecurityException, MqttException {
        isConnected = true;
    }

    @Override
    public IMqttToken connectWithResult(MqttConnectOptions options) throws MqttSecurityException, MqttException {
        isConnected = true;
        return null;
    }

    @Override
    public void disconnect() throws MqttException {
        isConnected = false;
    }

    @Override
    public void disconnect(long quiesceTimeout) throws MqttException {
        isConnected = false;
    }

    @Override
    public void disconnectForcibly() throws MqttException {
        isConnected = false;
    }

    @Override
    public void disconnectForcibly(long disconnectTimeout) throws MqttException {
        isConnected = false;
    }

    @Override
    public void disconnectForcibly(long quiesceTimeout, long disconnectTimeout) throws MqttException {
        isConnected = false;
    }

    @Override
    public void subscribe(String topicFilter) throws MqttException, MqttSecurityException {
        subscribedTopics.add(topicFilter);
    }

    @Override
    public void subscribe(String[] topicFilters) throws MqttException {
        subscribedTopics.addAll(Arrays.asList(topicFilters));
    }

    @Override
    public void subscribe(String topicFilter, int qos) throws MqttException {
        subscribedTopics.add(topicFilter);
    }

    @Override
    public void subscribe(String[] topicFilters, int[] qos) throws MqttException {
        subscribedTopics.addAll(Arrays.asList(topicFilters));
    }

    @Override
    public void subscribe(String topicFilter, IMqttMessageListener messageListener) throws MqttException, MqttSecurityException {
        subscribedTopics.add(topicFilter);
    }

    @Override
    public void subscribe(String[] topicFilters, IMqttMessageListener[] messageListeners) throws MqttException {
        subscribedTopics.addAll(Arrays.asList(topicFilters));
    }

    @Override
    public void subscribe(String topicFilter, int qos, IMqttMessageListener messageListener) throws MqttException {
        subscribedTopics.add(topicFilter);
    }

    @Override
    public void subscribe(String[] topicFilters, int[] qos, IMqttMessageListener[] messageListeners) throws MqttException {
        subscribedTopics.addAll(Arrays.asList(topicFilters));
    }

    @Override
    public void unsubscribe(String topicFilter) throws MqttException {

    }

    @Override
    public void unsubscribe(String[] topicFilters) throws MqttException {

    }

    @Override
    public void publish(String topic, byte[] payload, int qos, boolean retained) throws MqttException, MqttPersistenceException {
        lastPublishedTopic = topic;
        lastPublishedMessage = new String(payload, StandardCharsets.UTF_8);
    }

    @Override
    public void publish(String topic, MqttMessage message) throws MqttException, MqttPersistenceException {
        lastPublishedTopic = topic;
        lastPublishedMessage = new String(message.getPayload(), StandardCharsets.UTF_8);
    }

    @Override
    public void setCallback(MqttCallback callback) {

    }

    @Override
    public MqttTopic getTopic(String topic) {
        return null;
    }

    @Override
    public boolean isConnected() {
        return isConnected;
    }

    @Override
    public String getClientId() {
        return null;
    }

    @Override
    public String getServerURI() {
        return null;
    }

    @Override
    public IMqttDeliveryToken[] getPendingDeliveryTokens() {
        return new IMqttDeliveryToken[0];
    }

    @Override
    public void setManualAcks(boolean manualAcks) {

    }

    @Override
    public void messageArrivedComplete(int messageId, int qos) throws MqttException {

    }

    @Override
    public void close() throws MqttException {

    }
}
