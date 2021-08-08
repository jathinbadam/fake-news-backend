$("#seeAnotherField").change(function () {
  if ($(this).val() == "yes") {
    $("#otherFieldDiv").show();
    $("#otherField").attr("required", "");
    $("#otherField").attr("data-error", "This field is required.");
  } else {
    $("#otherFieldDiv").hide();
    $("#otherField").removeAttr("required");
    $("#otherField").removeAttr("data-error");
  }
});
$("#seeAnotherField").trigger("change");

$("#seeAnotherFieldGroup").change(function () {
  if ($(this).val() == "yes") {
    $("#otherFieldGroupDiv").show();
    $("#otherFieldGroupDiv2").hide();
    $("#otherField1").attr("required", "");
    $("#otherField1").attr("data-error", "This field is required.");
  } else {
    $("#otherFieldGroupDiv2").show();
    $("#otherFieldGroupDiv").hide();
    $("#otherField11").removeAttr("required");
    $("#otherField11").removeAttr("data-error");
  }
});
$("#seeAnotherFieldGroup").trigger("change");
