import tensorflow as tf

from transformers import TFBertModel


def TFKoBertModel(model, input_dim, output_dim):
    base_model = TFBertModel.from_pretrained(
        model, from_pt=True, cache_dir='cache/')

    input_token_ids = tf.keras.layers.Input(
        (input_dim,), dtype=tf.int32, name='input_token_ids')
    input_masks = tf.keras.layers.Input(
        (input_dim,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (input_dim,), dtype=tf.int32, name='input_segments')

    bert_outputs = base_model([input_token_ids, input_masks, input_segments])
    bert_outputs = bert_outputs[1]

    bert_outputs = tf.keras.layers.Dropout(0.2)(bert_outputs)
    final_output = tf.keras.layers.Dense(units=output_dim,
                                         activation=tf.nn.softmax,
                                         kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                             stddev=0.02),
                                         name='classifier')(bert_outputs)

    return tf.keras.Model(inputs=[input_token_ids, input_masks, input_segments],
                          outputs=final_output)


# class TFKoBertModel(tf.keras.Model):
#     def __init__(self, model):
#         super().__init__()
#         self.bert = TFBertModel.from_pretrained(
#             model, from_pt=True, cache_dir='cache/')

#         self.dropout = tf.keras.layers.Dropout(0.2)  # config.dropout
#         self.classifier = tf.keras.layers.Dense(units=2,  # config.output_dim,
#                                                 activation=tf.nn.softmax,
#                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(
#                                                     stddev=0.02),
#                                                 name='classifier')

#     def call(self, inputs, training=False):
#         input_ids, attention_mask, token_type_ids = inputs

#         x = self.bert(input_ids=input_ids,
#                       attention_mask=attention_mask,
#                       token_type_ids=token_type_ids)
#         x = x[1]  # (None, 768)

#         if training:
#             x = self.dropout(x, training=training)

#         x = self.classifier(x)

#         return x
