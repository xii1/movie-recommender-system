??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	?
*
dtype0
?
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K
*'
shared_nameembedding_1/embeddings
?
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes
:	?K
*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	?
*
dtype0
?
Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K
*.
shared_nameAdam/embedding_1/embeddings/m
?
1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m*
_output_shapes
:	?K
*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

: *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	?
*
dtype0
?
Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K
*.
shared_nameAdam/embedding_1/embeddings/v
?
1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v*
_output_shapes
:	?K
*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

: *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?<
value?<B?< B?<
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer_with_weights-4
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
 regularization_losses
!	keras_api
R
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
R
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
R
4	variables
5trainable_variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
R
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
?
Hiter

Ibeta_1

Jbeta_2
	Kdecay
Llearning_ratem?m?.m?/m?8m?9m?Bm?Cm?v?v?.v?/v?8v?9v?Bv?Cv?
8
0
1
.2
/3
84
95
B6
C7
8
0
1
.2
/3
84
95
B6
C7
 
?

Mlayers
Nlayer_metrics
	variables
trainable_variables
Onon_trainable_variables
Pmetrics
regularization_losses
Qlayer_regularization_losses
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?

Rlayers
Slayer_metrics
	variables
trainable_variables
Tnon_trainable_variables
Umetrics
regularization_losses
Vlayer_regularization_losses
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?

Wlayers
Xlayer_metrics
	variables
trainable_variables
Ynon_trainable_variables
Zmetrics
regularization_losses
[layer_regularization_losses
 
 
 
?

\layers
]layer_metrics
	variables
trainable_variables
^non_trainable_variables
_metrics
 regularization_losses
`layer_regularization_losses
 
 
 
?

alayers
blayer_metrics
"	variables
#trainable_variables
cnon_trainable_variables
dmetrics
$regularization_losses
elayer_regularization_losses
 
 
 
?

flayers
glayer_metrics
&	variables
'trainable_variables
hnon_trainable_variables
imetrics
(regularization_losses
jlayer_regularization_losses
 
 
 
?

klayers
llayer_metrics
*	variables
+trainable_variables
mnon_trainable_variables
nmetrics
,regularization_losses
olayer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
?

players
qlayer_metrics
0	variables
1trainable_variables
rnon_trainable_variables
smetrics
2regularization_losses
tlayer_regularization_losses
 
 
 
?

ulayers
vlayer_metrics
4	variables
5trainable_variables
wnon_trainable_variables
xmetrics
6regularization_losses
ylayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
?

zlayers
{layer_metrics
:	variables
;trainable_variables
|non_trainable_variables
}metrics
<regularization_losses
~layer_regularization_losses
 
 
 
?

layers
?layer_metrics
>	variables
?trainable_variables
?non_trainable_variables
?metrics
@regularization_losses
 ?layer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
?
?layers
?layer_metrics
D	variables
Etrainable_variables
?non_trainable_variables
?metrics
Fregularization_losses
 ?layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 

?0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
??
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_1/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/embedding_1/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2embedding_1/embeddingsembedding/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_64888
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_65374
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsembedding_1/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/embedding/embeddings/mAdam/embedding_1/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/embedding/embeddings/vAdam/embedding_1/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_65477??
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_64451

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????
2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_65215

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_645232
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
@__inference_model_layer_call_and_return_conditional_losses_64826
input_1
input_2$
embedding_1_64798:	?K
"
embedding_64801:	?

dense_64808: 
dense_64810: 
dense_1_64814: 
dense_1_64816:
dense_2_64820:
dense_2_64822:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_64798*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_644272%
#embedding_1/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_64801*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_644412#
!embedding/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_644512
flatten/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_644592
flatten_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_644682
concatenate/PartitionedCall?
dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_644752
dropout/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_64808dense_64810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_644882
dense/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_644992
dropout_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_64814dense_1_64816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_645122!
dense_1/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_645232
dropout_2/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_64820dense_2_64822*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_645362!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
@__inference_dense_layer_call_and_return_conditional_losses_64488

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_64523

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
@__inference_model_layer_call_and_return_conditional_losses_64979
inputs_0
inputs_15
"embedding_1_embedding_lookup_64937:	?K
3
 embedding_embedding_lookup_64943:	?
6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?embedding/embedding_lookup?embedding_1/embedding_lookupw
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_64937embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/64937*+
_output_shapes
:?????????
*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/64937*+
_output_shapes
:?????????
2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2)
'embedding_1/embedding_lookup/Identity_1s
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGather embedding_embedding_lookup_64943embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/64943*+
_output_shapes
:?????????
*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/64943*+
_output_shapes
:?????????
2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2'
%embedding/embedding_lookup/Identity_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
flatten/Const?
flatten/ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????
2
flatten/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
flatten_1/Const?
flatten_1/ReshapeReshape0embedding_1/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????
2
flatten_1/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate/concat
dropout/IdentityIdentityconcatenate/concat:output:0*
T0*'
_output_shapes
:?????????2
dropout/Identity?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

dense/Relu?
dropout_1/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
dropout_1/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Relu?
dropout_2/IdentityIdentitydense_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_2/Identity?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_2/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Sigmoid?
IdentityIdentitydense_2/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^embedding/embedding_lookup^embedding_1/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_65131

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_embedding_layer_call_fn_65054

inputs
unknown:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_644412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_64932
inputs_0
inputs_1
unknown:	?K

	unknown_0:	?

	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_647532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?-
?
@__inference_model_layer_call_and_return_conditional_losses_64543

inputs
inputs_1$
embedding_1_64428:	?K
"
embedding_64442:	?

dense_64489: 
dense_64491: 
dense_1_64513: 
dense_1_64515:
dense_2_64537:
dense_2_64539:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_1_64428*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_644272%
#embedding_1/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_64442*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_644412#
!embedding/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_644512
flatten/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_644592
flatten_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_644682
concatenate/PartitionedCall?
dropout/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_644752
dropout/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_64489dense_64491*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_644882
dense/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_644992
dropout_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_64513dense_1_64515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_645122!
dense_1/StatefulPartitionedCall?
dropout_2/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_645232
dropout_2/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_2_64537dense_2_64539*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_645362!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
F__inference_embedding_1_layer_call_and_return_conditional_losses_65081

inputs)
embedding_lookup_65075:	?K

identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_65075Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65075*+
_output_shapes
:?????????
*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/65075*+
_output_shapes
:?????????
2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_embedding_layer_call_and_return_conditional_losses_65064

inputs)
embedding_lookup_65058:	?

identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_65058Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65058*+
_output_shapes
:?????????
*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/65058*+
_output_shapes
:?????????
2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_embedding_layer_call_and_return_conditional_losses_64441

inputs)
embedding_lookup_64435:	?

identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_64435Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/64435*+
_output_shapes
:?????????
*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/64435*+
_output_shapes
:?????????
2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_embedding_1_layer_call_fn_65071

inputs
unknown:	?K

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_644272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_64512

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_65477
file_prefix8
%assignvariableop_embedding_embeddings:	?
<
)assignvariableop_1_embedding_1_embeddings:	?K
1
assignvariableop_2_dense_kernel: +
assignvariableop_3_dense_bias: 3
!assignvariableop_4_dense_1_kernel: -
assignvariableop_5_dense_1_bias:3
!assignvariableop_6_dense_2_kernel:-
assignvariableop_7_dense_2_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: B
/assignvariableop_15_adam_embedding_embeddings_m:	?
D
1assignvariableop_16_adam_embedding_1_embeddings_m:	?K
9
'assignvariableop_17_adam_dense_kernel_m: 3
%assignvariableop_18_adam_dense_bias_m: ;
)assignvariableop_19_adam_dense_1_kernel_m: 5
'assignvariableop_20_adam_dense_1_bias_m:;
)assignvariableop_21_adam_dense_2_kernel_m:5
'assignvariableop_22_adam_dense_2_bias_m:B
/assignvariableop_23_adam_embedding_embeddings_v:	?
D
1assignvariableop_24_adam_embedding_1_embeddings_v:	?K
9
'assignvariableop_25_adam_dense_kernel_v: 3
%assignvariableop_26_adam_dense_bias_v: ;
)assignvariableop_27_adam_dense_1_kernel_v: 5
'assignvariableop_28_adam_dense_1_bias_v:;
)assignvariableop_29_adam_dense_2_kernel_v:5
'assignvariableop_30_adam_dense_2_bias_v:
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_embedding_1_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_adam_embedding_embeddings_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp1assignvariableop_16_adam_embedding_1_embeddings_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_dense_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_embedding_embeddings_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp1assignvariableop_24_adam_embedding_1_embeddings_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31?
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
E
)__inference_dropout_1_layer_call_fn_65168

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_644992
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_65143

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_65163

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_2_layer_call_fn_65246

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_645362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_64459

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????
2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?2
?
@__inference_model_layer_call_and_return_conditional_losses_64753

inputs
inputs_1$
embedding_1_64725:	?K
"
embedding_64728:	?

dense_64735: 
dense_64737: 
dense_1_64741: 
dense_1_64743:
dense_2_64747:
dense_2_64749:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_1_64725*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_644272%
#embedding_1/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_64728*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_644412#
!embedding/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_644512
flatten/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_644592
flatten_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_644682
concatenate/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_646582!
dropout/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_64735dense_64737*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_644882
dense/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_646252#
!dropout_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_64741dense_1_64743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_645122!
dense_1/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_645922#
!dropout_2/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_64747dense_2_64749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_645362!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_1_layer_call_fn_65097

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_644592
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_65152

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_644882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_65121

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_644752
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
p
F__inference_concatenate_layer_call_and_return_conditional_losses_64468

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????
:?????????
:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_65173

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_646252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?2
?
@__inference_model_layer_call_and_return_conditional_losses_64858
input_1
input_2$
embedding_1_64830:	?K
"
embedding_64833:	?

dense_64840: 
dense_64842: 
dense_1_64846: 
dense_1_64848:
dense_2_64852:
dense_2_64854:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_64830*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_644272%
#embedding_1/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_64833*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_644412#
!embedding/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_644512
flatten/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_644592
flatten_1/PartitionedCall?
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_644682
concatenate/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_646582!
dropout/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_64840dense_64842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_644882
dense/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_646252#
!dropout_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_64846dense_1_64848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_645122!
dense_1/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_645922#
!dropout_2/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_64852dense_2_64854*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_645362!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_65190

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_64499

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_65210

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_65225

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_64888
input_1
input_2
unknown:	?K

	unknown_0:	?

	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_644082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?E
?
__inference__traced_save_65374
file_prefix3
/savev2_embedding_embeddings_read_readvariableop5
1savev2_embedding_1_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop1savev2_embedding_1_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?
:	?K
: : : :::: : : : : : : :	?
:	?K
: : : ::::	?
:	?K
: : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?
:%!

_output_shapes
:	?K
:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?
:%!

_output_shapes
:	?K
:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	?
:%!

_output_shapes
:	?K
:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
: 
?

?
%__inference_model_layer_call_fn_64794
input_1
input_2
unknown:	?K

	unknown_0:	?

	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_647532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
'__inference_dense_1_layer_call_fn_65199

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_645122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?@
?
 __inference__wrapped_model_64408
input_1
input_2;
(model_embedding_1_embedding_lookup_64366:	?K
9
&model_embedding_embedding_lookup_64372:	?
<
*model_dense_matmul_readvariableop_resource: 9
+model_dense_biasadd_readvariableop_resource: >
,model_dense_1_matmul_readvariableop_resource: ;
-model_dense_1_biasadd_readvariableop_resource:>
,model_dense_2_matmul_readvariableop_resource:;
-model_dense_2_biasadd_readvariableop_resource:
identity??"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?$model/dense_2/BiasAdd/ReadVariableOp?#model/dense_2/MatMul/ReadVariableOp? model/embedding/embedding_lookup?"model/embedding_1/embedding_lookup?
model/embedding_1/CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model/embedding_1/Cast?
"model/embedding_1/embedding_lookupResourceGather(model_embedding_1_embedding_lookup_64366model/embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@model/embedding_1/embedding_lookup/64366*+
_output_shapes
:?????????
*
dtype02$
"model/embedding_1/embedding_lookup?
+model/embedding_1/embedding_lookup/IdentityIdentity+model/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@model/embedding_1/embedding_lookup/64366*+
_output_shapes
:?????????
2-
+model/embedding_1/embedding_lookup/Identity?
-model/embedding_1/embedding_lookup/Identity_1Identity4model/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2/
-model/embedding_1/embedding_lookup/Identity_1~
model/embedding/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model/embedding/Cast?
 model/embedding/embedding_lookupResourceGather&model_embedding_embedding_lookup_64372model/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@model/embedding/embedding_lookup/64372*+
_output_shapes
:?????????
*
dtype02"
 model/embedding/embedding_lookup?
)model/embedding/embedding_lookup/IdentityIdentity)model/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@model/embedding/embedding_lookup/64372*+
_output_shapes
:?????????
2+
)model/embedding/embedding_lookup/Identity?
+model/embedding/embedding_lookup/Identity_1Identity2model/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2-
+model/embedding/embedding_lookup/Identity_1{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
model/flatten/Const?
model/flatten/ReshapeReshape4model/embedding/embedding_lookup/Identity_1:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:?????????
2
model/flatten/Reshape
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
model/flatten_1/Const?
model/flatten_1/ReshapeReshape6model/embedding_1/embedding_lookup/Identity_1:output:0model/flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????
2
model/flatten_1/Reshape?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2model/flatten/Reshape:output:0 model/flatten_1/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model/concatenate/concat?
model/dropout/IdentityIdentity!model/concatenate/concat:output:0*
T0*'
_output_shapes
:?????????2
model/dropout/Identity?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMulmodel/dropout/Identity:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model/dense/Relu?
model/dropout_1/IdentityIdentitymodel/dense/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
model/dropout_1/Identity?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMul!model/dropout_1/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_1/BiasAdd?
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense_1/Relu?
model/dropout_2/IdentityIdentity model/dense_1/Relu:activations:0*
T0*'
_output_shapes
:?????????2
model/dropout_2/Identity?
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#model/dense_2/MatMul/ReadVariableOp?
model/dense_2/MatMulMatMul!model/dropout_2/Identity:output:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_2/MatMul?
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_2/BiasAdd?
model/dense_2/SigmoidSigmoidmodel/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/dense_2/Sigmoid?
IdentityIdentitymodel/dense_2/Sigmoid:y:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp!^model/embedding/embedding_lookup#^model/embedding_1/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2D
 model/embedding/embedding_lookup model/embedding/embedding_lookup2H
"model/embedding_1/embedding_lookup"model/embedding_1/embedding_lookup:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
C
'__inference_flatten_layer_call_fn_65086

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_644512
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
F__inference_embedding_1_layer_call_and_return_conditional_losses_64427

inputs)
embedding_lookup_64421:	?K

identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_64421Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*)
_class
loc:@embedding_lookup/64421*+
_output_shapes
:?????????
*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*)
_class
loc:@embedding_lookup/64421*+
_output_shapes
:?????????
2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
W
+__inference_concatenate_layer_call_fn_65109
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_644682
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????
:?????????
:Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1
?
b
)__inference_dropout_2_layer_call_fn_65220

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_645922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_65103

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????
2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_65237

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_64910
inputs_0
inputs_1
unknown:	?K

	unknown_0:	?

	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_645432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?U
?
@__inference_model_layer_call_and_return_conditional_losses_65047
inputs_0
inputs_15
"embedding_1_embedding_lookup_64984:	?K
3
 embedding_embedding_lookup_64990:	?
6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?embedding/embedding_lookup?embedding_1/embedding_lookupw
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_64984embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/64984*+
_output_shapes
:?????????
*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/64984*+
_output_shapes
:?????????
2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2)
'embedding_1/embedding_lookup/Identity_1s
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGather embedding_embedding_lookup_64990embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/64990*+
_output_shapes
:?????????
*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/64990*+
_output_shapes
:?????????
2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????
2'
%embedding/embedding_lookup/Identity_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
flatten/Const?
flatten/ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????
2
flatten/Reshapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
flatten_1/Const?
flatten_1/ReshapeReshape0embedding_1/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:?????????
2
flatten_1/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate/concats
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const?
dropout/dropout/MulMulconcatenate/concat:output:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/dropout/Muly
dropout/dropout/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/dropout/Mul_1?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2

dense/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const?
dropout_1/dropout/MulMuldense/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout_1/dropout/Mulz
dropout_1/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout_1/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/Const?
dropout_2/dropout/MulMuldense_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_2/dropout/Mul|
dropout_2/dropout/ShapeShapedense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_2/dropout/Mul_1?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Sigmoid?
IdentityIdentitydense_2/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^embedding/embedding_lookup^embedding_1/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_64658

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_64562
input_1
input_2
unknown:	?K

	unknown_0:	?

	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_645432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_65092

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????
   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????
2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_64475

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
r
F__inference_concatenate_layer_call_and_return_conditional_losses_65116
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????
:?????????
:Q M
'
_output_shapes
:?????????

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_64625

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:????????? 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:????????? 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:????????? 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_65126

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_646582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_64592

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_65178

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:????????? 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_65257

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_64536

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
;
input_20
serving_default_input_2:0?????????;
dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?W
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer_with_weights-4
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?S
_tf_keras_network?S{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9724, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["embedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["embedding_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "shared_object_id": 21, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "input_2"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9724, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_1", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["embedding", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["embedding_1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_1", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 20}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9724, "output_dim": 10, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?
	variables
trainable_variables
 regularization_losses
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["embedding", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 24}}
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["embedding_1", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 25}}
?
&	variables
'trainable_variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["flatten", 0, 0, {}], ["flatten_1", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10]}, {"class_name": "TensorShape", "items": [null, 10]}]}
?
*	variables
+trainable_variables
,regularization_losses
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 9}
?

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?
4	variables
5trainable_variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 13}
?	

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 17}
?	

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
Hiter

Ibeta_1

Jbeta_2
	Kdecay
Llearning_ratem?m?.m?/m?8m?9m?Bm?Cm?v?v?.v?/v?8v?9v?Bv?Cv?"
	optimizer
X
0
1
.2
/3
84
95
B6
C7"
trackable_list_wrapper
X
0
1
.2
/3
84
95
B6
C7"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Mlayers
Nlayer_metrics
	variables
trainable_variables
Onon_trainable_variables
Pmetrics
regularization_losses
Qlayer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':%	?
2embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Rlayers
Slayer_metrics
	variables
trainable_variables
Tnon_trainable_variables
Umetrics
regularization_losses
Vlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'	?K
2embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Wlayers
Xlayer_metrics
	variables
trainable_variables
Ynon_trainable_variables
Zmetrics
regularization_losses
[layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

\layers
]layer_metrics
	variables
trainable_variables
^non_trainable_variables
_metrics
 regularization_losses
`layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

alayers
blayer_metrics
"	variables
#trainable_variables
cnon_trainable_variables
dmetrics
$regularization_losses
elayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

flayers
glayer_metrics
&	variables
'trainable_variables
hnon_trainable_variables
imetrics
(regularization_losses
jlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

klayers
llayer_metrics
*	variables
+trainable_variables
mnon_trainable_variables
nmetrics
,regularization_losses
olayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2dense/kernel
: 2
dense/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

players
qlayer_metrics
0	variables
1trainable_variables
rnon_trainable_variables
smetrics
2regularization_losses
tlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

ulayers
vlayer_metrics
4	variables
5trainable_variables
wnon_trainable_variables
xmetrics
6regularization_losses
ylayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_1/kernel
:2dense_1/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?

zlayers
{layer_metrics
:	variables
;trainable_variables
|non_trainable_variables
}metrics
<regularization_losses
~layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

layers
?layer_metrics
>	variables
?trainable_variables
?non_trainable_variables
?metrics
@regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?layer_metrics
D	variables
Etrainable_variables
?non_trainable_variables
?metrics
Fregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 29}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:*	?
2Adam/embedding/embeddings/m
.:,	?K
2Adam/embedding_1/embeddings/m
#:! 2Adam/dense/kernel/m
: 2Adam/dense/bias/m
%:# 2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
,:*	?
2Adam/embedding/embeddings/v
.:,	?K
2Adam/embedding_1/embeddings/v
#:! 2Adam/dense/kernel/v
: 2Adam/dense/bias/v
%:# 2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
?2?
%__inference_model_layer_call_fn_64562
%__inference_model_layer_call_fn_64910
%__inference_model_layer_call_fn_64932
%__inference_model_layer_call_fn_64794?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_model_layer_call_and_return_conditional_losses_64979
@__inference_model_layer_call_and_return_conditional_losses_65047
@__inference_model_layer_call_and_return_conditional_losses_64826
@__inference_model_layer_call_and_return_conditional_losses_64858?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_64408?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *N?K
I?F
!?
input_1?????????
!?
input_2?????????
?2?
)__inference_embedding_layer_call_fn_65054?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_embedding_layer_call_and_return_conditional_losses_65064?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_embedding_1_layer_call_fn_65071?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_embedding_1_layer_call_and_return_conditional_losses_65081?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_65086?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_65092?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_1_layer_call_fn_65097?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_1_layer_call_and_return_conditional_losses_65103?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_65109?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_65116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_65121
'__inference_dropout_layer_call_fn_65126?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_65131
B__inference_dropout_layer_call_and_return_conditional_losses_65143?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_65152?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_65163?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dropout_1_layer_call_fn_65168
)__inference_dropout_1_layer_call_fn_65173?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_65178
D__inference_dropout_1_layer_call_and_return_conditional_losses_65190?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_65199?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_65210?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dropout_2_layer_call_fn_65215
)__inference_dropout_2_layer_call_fn_65220?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_2_layer_call_and_return_conditional_losses_65225
D__inference_dropout_2_layer_call_and_return_conditional_losses_65237?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_65246?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_65257?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_64888input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_64408?./89BCX?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "1?.
,
dense_2!?
dense_2??????????
F__inference_concatenate_layer_call_and_return_conditional_losses_65116?Z?W
P?M
K?H
"?
inputs/0?????????

"?
inputs/1?????????

? "%?"
?
0?????????
? ?
+__inference_concatenate_layer_call_fn_65109vZ?W
P?M
K?H
"?
inputs/0?????????

"?
inputs/1?????????

? "???????????
B__inference_dense_1_layer_call_and_return_conditional_losses_65210\89/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_65199O89/?,
%?"
 ?
inputs????????? 
? "???????????
B__inference_dense_2_layer_call_and_return_conditional_losses_65257\BC/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_2_layer_call_fn_65246OBC/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_65163\.//?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? x
%__inference_dense_layer_call_fn_65152O.//?,
%?"
 ?
inputs?????????
? "?????????? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_65178\3?0
)?&
 ?
inputs????????? 
p 
? "%?"
?
0????????? 
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_65190\3?0
)?&
 ?
inputs????????? 
p
? "%?"
?
0????????? 
? |
)__inference_dropout_1_layer_call_fn_65168O3?0
)?&
 ?
inputs????????? 
p 
? "?????????? |
)__inference_dropout_1_layer_call_fn_65173O3?0
)?&
 ?
inputs????????? 
p
? "?????????? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_65225\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_65237\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? |
)__inference_dropout_2_layer_call_fn_65215O3?0
)?&
 ?
inputs?????????
p 
? "??????????|
)__inference_dropout_2_layer_call_fn_65220O3?0
)?&
 ?
inputs?????????
p
? "???????????
B__inference_dropout_layer_call_and_return_conditional_losses_65131\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_65143\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? z
'__inference_dropout_layer_call_fn_65121O3?0
)?&
 ?
inputs?????????
p 
? "??????????z
'__inference_dropout_layer_call_fn_65126O3?0
)?&
 ?
inputs?????????
p
? "???????????
F__inference_embedding_1_layer_call_and_return_conditional_losses_65081_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????

? ?
+__inference_embedding_1_layer_call_fn_65071R/?,
%?"
 ?
inputs?????????
? "??????????
?
D__inference_embedding_layer_call_and_return_conditional_losses_65064_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????

? 
)__inference_embedding_layer_call_fn_65054R/?,
%?"
 ?
inputs?????????
? "??????????
?
D__inference_flatten_1_layer_call_and_return_conditional_losses_65103\3?0
)?&
$?!
inputs?????????

? "%?"
?
0?????????

? |
)__inference_flatten_1_layer_call_fn_65097O3?0
)?&
$?!
inputs?????????

? "??????????
?
B__inference_flatten_layer_call_and_return_conditional_losses_65092\3?0
)?&
$?!
inputs?????????

? "%?"
?
0?????????

? z
'__inference_flatten_layer_call_fn_65086O3?0
)?&
$?!
inputs?????????

? "??????????
?
@__inference_model_layer_call_and_return_conditional_losses_64826?./89BC`?]
V?S
I?F
!?
input_1?????????
!?
input_2?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_64858?./89BC`?]
V?S
I?F
!?
input_1?????????
!?
input_2?????????
p

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_64979?./89BCb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_65047?./89BCb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
%__inference_model_layer_call_fn_64562?./89BC`?]
V?S
I?F
!?
input_1?????????
!?
input_2?????????
p 

 
? "???????????
%__inference_model_layer_call_fn_64794?./89BC`?]
V?S
I?F
!?
input_1?????????
!?
input_2?????????
p

 
? "???????????
%__inference_model_layer_call_fn_64910?./89BCb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
%__inference_model_layer_call_fn_64932?./89BCb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
#__inference_signature_wrapper_64888?./89BCi?f
? 
_?\
,
input_1!?
input_1?????????
,
input_2!?
input_2?????????"1?.
,
dense_2!?
dense_2?????????