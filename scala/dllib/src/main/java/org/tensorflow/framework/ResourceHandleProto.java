// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorboard/src/resource_handle.proto

package org.tensorflow.framework;

public final class ResourceHandleProto {
  private ResourceHandleProto() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_tensorboard_ResourceHandle_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_tensorboard_ResourceHandle_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n%tensorboard/src/resource_handle.proto\022" +
      "\013tensorboard\"m\n\016ResourceHandle\022\016\n\006device" +
      "\030\001 \001(\t\022\021\n\tcontainer\030\002 \001(\t\022\014\n\004name\030\003 \001(\t\022" +
      "\021\n\thash_code\030\004 \001(\004\022\027\n\017maybe_type_name\030\005 " +
      "\001(\tB4\n\030org.org.org.tensorflow.frameworkB\023Resourc" +
      "eHandleProtoP\001\370\001\001b\006proto3"
    };
    com.google.protobuf.Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
        new com.google.protobuf.Descriptors.FileDescriptor.    InternalDescriptorAssigner() {
          public com.google.protobuf.ExtensionRegistry assignDescriptors(
              com.google.protobuf.Descriptors.FileDescriptor root) {
            descriptor = root;
            return null;
          }
        };
    com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
        }, assigner);
    internal_static_tensorboard_ResourceHandle_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_tensorboard_ResourceHandle_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_tensorboard_ResourceHandle_descriptor,
        new java.lang.String[] { "Device", "Container", "Name", "HashCode", "MaybeTypeName", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}
